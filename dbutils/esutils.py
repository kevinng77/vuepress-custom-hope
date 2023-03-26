# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
from elasticsearch.helpers import bulk
from typing import Any, Dict, Generator, List, Optional, Type, Union
from file_converter.schema import Document
from tqdm import tqdm
import logging
from itertools import islice
from scipy.special import expit

import numpy as np

import logging

logger = logging.getLogger(__name__)

def get_batches_from_generator(iterable, n):
    """
    Batch elements of an iterable into fixed-length chunks or blocks.
    """
    it = iter(iterable)
    x = tuple(islice(it, n))
    while x:
        yield x
        x = tuple(islice(it, n))

class EsUtils(object):
    def __init__(self,
                 host: str = "localhost",
                 port: int = 9200,
                 recreate_index: bool = False,
                 create_index: bool = True,
                 index: str = "document",
                 search_fields: list = ["content", "name"],
                 label_index: str = "label",
                 embedding_dim: int = 324,
                 custom_mapping: dict = None
                 ) -> None:

        self.client = Elasticsearch(f"http://{host}:{port}",basic_auth=('elastic', '123456'))
        self.search_fields = search_fields
        self.embedding_field = "embedding"
        self.name_field = "name"
        self.content_field = "content"
        self.analyzer = "standard"
        self.embedding_dim = embedding_dim
        self.custom_mapping = custom_mapping
        if recreate_index:
            self.delete_index(index)
            self.delete_index(label_index)
            self._create_document_index(index)
            self._create_label_index(label_index)
        elif create_index:
            self._create_document_index(index)
            self._create_label_index(label_index)
        self.duplicate_documents = "overwrite"  # ["overwrite", "skip"]
        self.index=index
        self.refresh_type = "wait_for"

    def delete_index(self, index: str):
        """
        Delete an existing elasticsearch index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        self.client.indices.delete(index=index, ignore=[400, 404])
        logger.debug(f"deleted elasticsearch index {index}")

    def _create_document_index(self, index_name: str):
        """
        Create a new index for storing documents. In case if an index with the name already exists, it ensures that
        the embedding_field is present.
        """
        # check if the existing index has the embedding field; if not create it
        if self.client.indices.exists(index=index_name):
            mapping = self.client.indices.get(index_name)[
                index_name]["mappings"]
            if self.search_fields:
                for search_field in self.search_fields:
                    if search_field in mapping["properties"] and mapping["properties"][search_field]["type"] != "text":
                        raise Exception(
                            f"The search_field '{search_field}' of index '{index_name}' with type '{mapping['properties'][search_field]['type']}' "
                            f"does not have the right type 'text' to be queried in fulltext search. Please use only 'text' type properties as search_fields. "
                            f"This error might occur if you are trying to use pipelines 1.0 and above with an existing elasticsearch index created with a previous version of pipelines."
                            f"In this case deleting the index with `curl -X DELETE \"{self.pipeline_config['params']['host']}:{self.pipeline_config['params']['port']}/{index_name}\"` will fix your environment. "
                            f"Note, that all data stored in the index will be lost!"
                        )
            if self.embedding_field:
                if (
                    self.embedding_field in mapping["properties"]
                    and mapping["properties"][self.embedding_field]["type"] != "dense_vector"
                ):
                    raise Exception(
                        f"The '{index_name}' index in Elasticsearch already has a field called '{self.embedding_field}'"
                        f" with the type '{mapping['properties'][self.embedding_field]['type']}'. Please update the "
                        f"document_store to use a different name for the embedding_field parameter."
                    )
                mapping["properties"][self.embedding_field] = {
                    "type": "dense_vector", "dims": self.embedding_dim}
                self.client.indices.put_mapping(
                    index=index_name, body=mapping)
            return

        if self.custom_mapping:
            mapping = self.custom_mapping
        else:
            mapping = {
                "mappings": {
                    "properties": {self.name_field: {"type": "keyword"},
                                   self.content_field: {"type": "text"}},
                    "dynamic_templates": [
                        {
                            "strings": {
                                "path_match": "*",
                                "match_mapping_type": "string",
                                "mapping": {"type": "keyword"},
                            }
                        }
                    ],
                },
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": self.analyzer,
                            }
                        }
                    }
                },
            }

            for field in self.search_fields:
                mapping["mappings"]["properties"].update(
                    {field: {"type": "text"}})

            if self.embedding_field:
                mapping["mappings"]["properties"][self.embedding_field] = {
                    "type": "dense_vector",
                    "dims": self.embedding_dim,
                }

        try:
            self.client.indices.create(
                index=index_name, body=mapping)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name):
                raise e

    def _create_label_index(self, index_name: str):
        if self.client.indices.exists(index=index_name):
            return
        mapping = {
            "mappings": {
                "properties": {
                    "query": {"type": "text"},
                    "answer": {"type": "flattened"},  # light-weight but less search options than full object
                    "document": {"type": "flattened"},
                    "is_correct_answer": {"type": "boolean"},
                    "is_correct_document": {"type": "boolean"},
                    "origin": {"type": "keyword"},  # e.g. user-feedback or gold-label
                    "document_id": {"type": "keyword"},
                    "no_answer": {"type": "boolean"},
                    "pipeline_id": {"type": "keyword"},
                    "created_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"},
                    "updated_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"}
                    # TODO add pipeline_hash and pipeline_name once we migrated the REST API to pipelines
                }
            }
        }
        try:
            self.client.indices.create(index=index_name, body=mapping)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name):
                raise e

    def _convert_es_hit_to_document(
        self,
        hit: dict,
        return_embedding: bool,
        adapt_score_for_embedding: bool = False,
    ) -> Document:
        # We put all additional data of the doc into meta_data and return it in the API
        meta_data = {
            k: v
            for k, v in hit["_source"].items()
            if k not in (self.content_field, "content_type", self.embedding_field)
        }
        name = meta_data.pop(self.name_field, None)
        if name:
            meta_data["name"] = name

        if "highlight" in hit:
            meta_data["highlighted"] = hit["highlight"]

        score = hit["_score"]
        if score:
            if adapt_score_for_embedding:
                score = self._scale_embedding_score(score)
                if self.similarity == "cosine":
                    score = (score + 1) / 2  # scaling probability from cosine similarity
                else:
                    score = float(expit(np.asarray(score / 100)))  # scaling probability from dot product and l2
            else:
                score = float(expit(np.asarray(score / 8)))  # scaling probability from TFIDF/BM25

        embedding = None
        if return_embedding:
            embedding_list = hit["_source"].get(self.embedding_field)
            if embedding_list:
                embedding = np.asarray(embedding_list, dtype=np.float32)

        doc_dict = {
            "id": hit["_id"],
            "content": hit["_source"].get(self.content_field),
            "content_type": hit["_source"].get("content_type", None),
            "meta": meta_data,
            "es_ann_score": score,
            "score": score,
            "embedding": embedding,
        }
        document = Document.from_dict(doc_dict)

        return document

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
    ):
        """
        Indexes documents for later queries in Elasticsearch.

        Behaviour if a document with the same ID already exists in ElasticSearch:
        a) (Default) Throw Elastic's standard error message for duplicate IDs.
        b) If `self.update_existing_documents=True` for DocumentStore: Overwrite existing documents.
        (This is only relevant if you pass your own ID when initializing a `Document`.
        If don't set custom IDs for your Documents or just pass a list of dictionaries here,
        they will automatically get UUIDs assigned. See the `Document` class for details)

        :param documents: a list of Python dictionaries or a list of pipelines Document objects.
                          For documents as dictionaries, the format is {"content": "<the-actual-text>"}.
                          Optionally: Include meta data via {"content": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
                          Advanced: If you are using your own Elasticsearch mapping, the key names in the dictionary
                          should be changed to what you have set for self.content_field and self.name_field.
        :param index: Elasticsearch index where the documents should be indexed. If not supplied, self.index will be used.
        :param batch_size: Number of documents that are passed to Elasticsearch's bulk function at a time.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return: None
        """

        if index and not self.client.indices.exists(index=index):
            self._create_document_index(index)

        if index is None:
            index = self.index

        field_map = {self.content_field: "content", self.embedding_field: "embedding"}
        document_objects = [
            Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents
        ]

        documents_to_index = []
        for doc in document_objects:
            _doc = {
                # create : 不指定 tag 文档id 时，创建新文档。
                # index : 
                "_op_type": "index" if self.duplicate_documents == "overwrite" else "create",
                "_index": index,
                **doc.to_dict(field_map=field_map),
            }  # type: Dict[str, Any]

            # cast embedding type as ES cannot deal with np.array
            if _doc[self.embedding_field] is not None:
                if type(_doc[self.embedding_field]) == np.ndarray:
                    _doc[self.embedding_field] = _doc[self.embedding_field].tolist()

            # rename id for elastic
            _doc["_id"] = str(_doc.pop("id"))

            # don't index query score and empty fields
            _ = _doc.pop("score", None)
            _doc = {k: v for k, v in _doc.items() if v is not None}

            # In order to have a flat structure in elastic + similar behaviour to the other DocumentStores,
            # we "unnest" all value within "meta"
            if "meta" in _doc.keys():
                for k, v in _doc["meta"].items():
                    _doc[k] = v
                _doc.pop("meta")
            documents_to_index.append(_doc)

            # Pass batch_size number of documents to bulk
            if len(documents_to_index) % batch_size == 0:
                bulk(self.client, documents_to_index, request_timeout=300, refresh=self.refresh_type)
                documents_to_index = []

        if documents_to_index:
            bulk(self.client, documents_to_index, request_timeout=300, refresh=self.refresh_type)

    def update_embeddings(
        self,
        retriever,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        update_existing_embeddings: bool = True,
        batch_size: int = 10_000,
    ):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to update the embeddings.
        :param index: Index name to update
        :param update_existing_embeddings: Whether to update existing embeddings of the documents. If set to False,
                                           only documents without embeddings are processed. This mode can be used for
                                           incremental updating of embeddings, wherein, only newly indexed documents
                                           get processed.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            ```
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :return: None
        """
        if index is None:
            index = self.index

        if self.refresh_type == "false":
            self.client.indices.refresh(index=index)

        if not self.embedding_field:
            raise RuntimeError("Specify the arg `embedding_field` when initializing ElasticsearchDocumentStore()")

        if update_existing_embeddings:
            document_count = self.get_document_count(index=index)
            logger.info(f"Updating embeddings for all {document_count} docs ...")
        else:
            document_count = self.get_document_count(
                index=index, filters=filters, only_documents_without_embedding=True
            )
            logger.info(f"Updating embeddings for {document_count} docs without embeddings ...")

        result = self._get_all_documents_in_index(
            index=index,
            filters=filters,
            batch_size=batch_size,
            only_documents_without_embedding=not update_existing_embeddings,
        )

        logging.getLogger("elasticsearch").setLevel(logging.CRITICAL)

        with tqdm(total=document_count, position=0, unit=" Docs", desc="Updating embeddings") as progress_bar:
            for result_batch in get_batches_from_generator(result, batch_size):
                document_batch = [
                    # List[Document]
                    self._convert_es_hit_to_document(hit, return_embedding=False) for hit in result_batch
                ]

                # TODO embedding model here
                embeddings = retriever.embed_documents(document_batch)  # type: ignore


                assert len(document_batch) == len(embeddings)

                if embeddings[0].shape[0] != self.embedding_dim:
                    raise RuntimeError(
                        f"Embedding dim. of model ({embeddings[0].shape[0]})"
                        f" doesn't match embedding dim. in DocumentStore ({self.embedding_dim})."
                        "Specify the arg `embedding_dim` when initializing ElasticsearchDocumentStore()"
                    )
                doc_updates = []
                for doc, emb in zip(document_batch, embeddings):
                    update = {
                        "_op_type": "update",
                        "_index": index,
                        "_id": doc.id,
                        "doc": {self.embedding_field: emb.tolist()},
                    }
                    doc_updates.append(update)

                bulk(self.client, doc_updates, request_timeout=300, refresh=self.refresh_type)
                progress_bar.update(batch_size)
