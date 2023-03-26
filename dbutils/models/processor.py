from typing import Optional,  List
import random
import logging
from typing import Union, Optional, List

logger = logging.getLogger(__name__)

class SampleBasket:
    """An object that contains one source text and the one or more samples that will be processed. This
    is needed for tasks like question answering where the source text can generate multiple input - label
    pairs."""

    def __init__(
        self,
        id_internal: Optional[Union[int, str]],
        raw: dict,
        id_external: str = None,
        samples= None,
    ):
        """
        :param id_internal: A unique identifying id. Used for identification within pipelines.
        :param external_id: Used for identification outside of pipelines. E.g. if another framework wants to pass along its own id with the results.
        :param raw: Contains the various data needed to form a sample. It is ideally in human readable form.
        :param samples(Optional[List[Sample]]) : An optional list of Samples used to populate the basket at initialization.
        """
        self.id_internal = id_internal
        self.id_external = id_external
        self.raw = raw
        self.samples = samples


class TextSimilarityProcessor:
    """
    Used to handle the Dense Passage Retrieval datasets that come in json format, example: biencoder-nq-train.json, biencoder-nq-dev.json, trivia-train.json, trivia-dev.json

    dataset format: list of dictionaries with keys: 'dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'
    Each sample is a dictionary of format:
    {"dataset": str,
    "question": str,
    "answers": list of str
    "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    }

    """

    def __init__(
        self,
        query_tokenizer,  # type: ignore
        passage_tokenizer,  # type: ignore
        max_seq_len_query: int,
        max_seq_len_passage: int,
        data_dir: str = "",
        metric=None,  # type: ignore
        train_filename: str = "train.json",
        dev_filename: Optional[str] = None,
        test_filename: Optional[str] = "test.json",
        dev_split: float = 0.1,
        proxies: Optional[dict] = None,
        max_samples: Optional[int] = None,
        embed_title: bool = True,
        num_positives: int = 1,
        num_hard_negatives: int = 1,
        shuffle_negatives: bool = True,
        shuffle_positives: bool = False,
        label_list: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        :param query_tokenizer: Used to split a question (str) into tokens
        :param passage_tokenizer: Used to split a passage (str) into tokens.
        :param max_seq_len_query: Query samples are truncated after this many tokens.
        :param max_seq_len_passage: Context/Passage Samples are truncated after this many tokens.
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `pipelines.basics.data_handler.utils`_.
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: None
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :param max_samples: maximum number of samples to use
        :param embed_title: Whether to embed title in passages during tensorization (bool),
        :param num_hard_negatives: maximum number to hard negative context passages in a sample
        :param num_positives: maximum number to positive context passages in a sample
        :param shuffle_negatives: Whether to shuffle all the hard_negative passages before selecting the num_hard_negative number of passages
        :param shuffle_positives: Whether to shuffle all the positive passages before selecting the num_positive number of passages
        :param label_list: list of labels to predict. Usually ["hard_negative", "positive"]
        :param kwargs: placeholder for passing generic parameters
        """
        # TODO If an arg is misspelt, e.g. metrics, it will be swallowed silently by kwargs

        # Custom processor attributes
        self.max_samples = max_samples
        self.query_tokenizer = query_tokenizer
        self.passage_tokenizer = passage_tokenizer
        self.embed_title = embed_title
        self.num_hard_negatives = num_hard_negatives
        self.num_positives = num_positives
        self.shuffle_negatives = shuffle_negatives
        self.shuffle_positives = shuffle_positives
        self.max_seq_len_query = max_seq_len_query
        self.max_seq_len_passage = max_seq_len_passage

        super(TextSimilarityProcessor, self).__init__(
            tokenizer=None,  # type: ignore
            max_seq_len=0,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        if metric:
            self.add_task(
                name="text_similarity",
                metric=metric,
                label_list=label_list,
                label_name="label",
                task_type="text_similarity",
            )
        else:
            logger.info(
                "Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                "using the default task or add a custom task later via processor.add_task()"
            )

    def dataset_from_dicts(self, dicts: List[dict], indices: Optional[List[int]] = None, return_baskets: bool = False):
        """
        Convert input dictionaries into a paddle dataset for TextSimilarity.
        For conversion we have an internal representation called "baskets".
        Each basket is one query and related text passages (positive passages fitting to the query and negative
        passages that do not fit the query)
        Each stage adds or transforms specific information to our baskets.

        :param dicts: input dictionary with DPR-style content
                        {"query": str,
                         "passages": List[
                                        {'title': str,
                                        'text': str,
                                        'label': 'hard_negative',
                                        'external_id': str},
                                        ....
                                        ]
                         }
        :param indices: indices used during multiprocessing so that IDs assigned to our baskets is unique
        :param return_baskets: whether to return the baskets or not (baskets are needed during inference)
        :return: dataset, tensor_names, problematic_ids, [baskets]
        """
        # Take the dict and insert into our basket structure, this stages also adds an internal IDs
        baskets = self._fill_baskets(dicts, indices)

        # Separat conversion of query
        baskets = self._convert_queries(baskets=baskets)

        # and context passages. When converting the context the label is also assigned.
        baskets = self._convert_contexts(baskets=baskets)

        # Convert features into paddle dataset, this step also removes and logs potential errors during preprocessing
        dataset, tensor_names, problematic_ids, baskets = self._create_dataset(baskets)

        if problematic_ids:
            logger.error(
                f"There were {len(problematic_ids)} errors during preprocessing at positions: {problematic_ids}"
            )

        if return_baskets:
            return dataset, tensor_names, problematic_ids, baskets
        else:
            return dataset, tensor_names, problematic_ids

    def _fill_baskets(self, dicts: List[dict], indices: Optional[List[int]]):
        baskets = []
        if not indices:
            indices = list(range(len(dicts)))
        for d, id_internal in zip(dicts, indices):
            basket = SampleBasket(id_external=None, id_internal=id_internal, raw=d)
            baskets.append(basket)
        return baskets

    def _convert_queries(self, baskets):
        """
        Args:
            baskets: List[SampleBasket]
        """
        for basket in baskets:
            clear_text = {}
            tokenized = {}
            features = [{}]  # type: ignore
            # extract query, positive context passages and titles, hard-negative passages and titles
            if "query" in basket.raw:
                try:
                    query = self._normalize_question(basket.raw["query"])
                    query_inputs = self.query_tokenizer(text=query, max_seq_len=self.max_seq_len_query)
                    tokenized_query = self.query_tokenizer.convert_ids_to_tokens(query_inputs["input_ids"])

                    if len(tokenized_query) == 0:
                        logger.warning(
                            f"The query could not be tokenized, likely because it contains a character that the query tokenizer does not recognize"
                        )
                        return None

                    clear_text["query_text"] = query
                    tokenized["query_tokens"] = tokenized_query
                    features[0]["query_input_ids"] = query_inputs["input_ids"]
                    features[0]["query_segment_ids"] = query_inputs["token_type_ids"]
                except Exception as e:
                    features = None  # type: ignore

            sample = Sample(id="", clear_text=clear_text, tokenized=tokenized, features=features)  # type: ignore
            basket.samples = [sample]
        return baskets

    def _convert_contexts(self, baskets):
        """
        Args:
            baskets: List[SampleBasket]
        """
        for basket in baskets:
            if "passages" in basket.raw:
                try:
                    positive_context = list(filter(lambda x: x["label"] == "positive", basket.raw["passages"]))
                    if self.shuffle_positives:
                        random.shuffle(positive_context)
                    positive_context = positive_context[: self.num_positives]
                    hard_negative_context = list(
                        filter(lambda x: x["label"] == "hard_negative", basket.raw["passages"])
                    )
                    if self.shuffle_negatives:
                        random.shuffle(hard_negative_context)
                    hard_negative_context = hard_negative_context[: self.num_hard_negatives]

                    positive_ctx_titles = [passage.get("title", None) for passage in positive_context]
                    positive_ctx_texts = [passage["text"] for passage in positive_context]
                    hard_negative_ctx_titles = [passage.get("title", None) for passage in hard_negative_context]
                    hard_negative_ctx_texts = [passage["text"] for passage in hard_negative_context]

                    # all context passages and labels: 1 for positive context and 0 for hard-negative context
                    ctx_label = [1] * self.num_positives + [0] * self.num_hard_negatives
                    # featurize context passages
                    if self.embed_title:
                        # concatenate title with positive context passages + negative context passages
                        all_ctx = self._combine_title_context(
                            positive_ctx_titles, positive_ctx_texts
                        ) + self._combine_title_context(hard_negative_ctx_titles, hard_negative_ctx_texts)
                    else:
                        all_ctx = positive_ctx_texts + hard_negative_ctx_texts

                    # assign empty string tuples if hard_negative passages less than num_hard_negatives
                    all_ctx += [("", "")] * ((self.num_positives + self.num_hard_negatives) - len(all_ctx))

                    # [text] -> tokenize -> id
                    ctx_inputs = self.passage_tokenizer(text=all_ctx[0], max_seq_len=self.max_seq_len_passage)

                    # get tokens in string format
                    tokenized_passage = [
                        self.passage_tokenizer.convert_ids_to_tokens(ctx) for ctx in ctx_inputs["input_ids"]
                    ]
                    # we only have one sample containing query and corresponding (multiple) context features
                    sample = basket.samples[0]  # type: ignore
                    sample.clear_text["passages"] = positive_context + hard_negative_context
                    sample.tokenized["passages_tokens"] = tokenized_passage  # type: ignore
                    sample.features[0]["passage_input_ids"] = ctx_inputs["input_ids"]  # type: ignore
                    sample.features[0]["passage_segment_ids"] = ctx_inputs["token_type_ids"]  # type: ignore
                except Exception as e:
                    basket.samples[0].features = None  # type: ignore

        return baskets

    def _create_dataset(self, baskets):
        """
        Convert python features into paddle dataset.
        Also removes potential errors during preprocessing.
        Flattens nested basket structure to create a flat list of features
        Args:
            baskets: List[SampleBasket]
        """
        features_flat: List[dict] = []
        basket_to_remove = []
        problematic_ids: set = set()
        for basket in baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:  # type: ignore
                    features_flat.extend(sample.features)  # type: ignore
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        if len(basket_to_remove) > 0:
            for basket in basket_to_remove:
                # if basket_to_remove is not empty remove the related baskets
                problematic_ids.add(basket.id_internal)
                baskets.remove(basket)

        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names, problematic_ids, baskets

    @staticmethod
    def _normalize_question(question: str) -> str:
        """Removes '?' from queries/questions"""
        if question[-1] == "?":
            question = question[:-1]
        return question

    @staticmethod
    def _combine_title_context(titles: List[str], texts: List[str]):
        res = []
        for title, ctx in zip(titles, texts):
            if title is None:
                title = ""
                logger.warning(
                    f"Couldn't find title although `embed_title` is set to True. Using title='' now. Related passage text: '{ctx}' "
                )
            res.append(tuple((title, ctx)))
        return res
