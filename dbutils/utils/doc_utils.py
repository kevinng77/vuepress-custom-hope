# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Callable, Dict, List, Optional

import logging
from pathlib import Path

from file_converter import (
    BaseConverter,
    DocxToTextConverter,
    PDFToTextConverter,
    TextConverter,
    # ImageToTextConverter,
    MarkdownConverter
)

logger = logging.getLogger(__name__)


def convert_files_to_dicts(
    dir_path: str,
    clean_func: Optional[Callable] = None,
    split_paragraphs: bool = False,
    split_answers: bool = False,
    encoding: Optional[str] = None,
) -> List[dict]:
    """
    Convert all files(.txt, .pdf, .docx) in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.

    :param dir_path: path for the documents to be written to the DocumentStore
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param split_paragraphs: split text in paragraphs.
    :param split_answers: split text into two columns, including question column, answer column.
    :param encoding: character encoding to use when converting pdf documents.
    """
    file_paths = [p for p in Path(dir_path).glob("**/*")]
    # allowed_suffixes = [".pdf", ".txt", ".docx", ".png", ".jpg", ".md"]
    allowed_suffixes = [ ".md"]

    suffix2converter: Dict[str, BaseConverter] = {}

    suffix2paths: Dict[str, List[Path]] = {}
    for path in file_paths:
        file_suffix = path.suffix.lower()
        if file_suffix in allowed_suffixes:
            if file_suffix not in suffix2paths:
                suffix2paths[file_suffix] = []
            suffix2paths[file_suffix].append(path)
        elif not path.is_dir():
            logger.warning(
                "Skipped file {0} as type {1} is not supported here. "
                "See pipelines.file_converter for support of more file types".format(path, file_suffix)
            )

    # No need to initialize converter if file type not present
    for file_suffix in suffix2paths.keys():
        # if file_suffix == ".pdf":
        #     suffix2converter[file_suffix] = PDFToTextConverter()
        if file_suffix == ".txt":
            suffix2converter[file_suffix] = TextConverter()
        # if file_suffix == ".docx":
        #     suffix2converter[file_suffix] = DocxToTextConverter()
        # if file_suffix == ".png" or file_suffix == ".jpg":
        #     suffix2converter[file_suffix] = ImageToTextConverter()
        if file_suffix == ".md":
            suffix2converter[file_suffix] = MarkdownConverter()
        else:
            logger.info(f"Unsupport Type [{file_suffix}] .")
    documents = []
    for suffix, paths in suffix2paths.items():
        for path in paths:
            if encoding is None and suffix == ".pdf":
                encoding = "Latin1"
            logger.info("Converting {}".format(path))
            list_documents = suffix2converter[suffix].convert(
                file_path=path,
                meta=None,
                encoding=encoding,
            )  # PDFToTextConverter, TextConverter, ImageToTextConverter and DocxToTextConverter return a list containing a single dict
            for document in list_documents:
                text = document["content"]

                if clean_func:
                    text = clean_func(text)

                if split_paragraphs:
                    # TODO 确认分段逻辑
                    for para in text.split("\n"):
                        if not para.strip():  # skip empty paragraphs
                            continue
                        if split_answers:
                            query, answer = para.split("\t")
                            meta_data = {"name": path.name, "answer": answer}
                            # Add image list parsed from docx into meta
                            if document["meta"] is not None and "images" in document["meta"]:
                                meta_data["images"] = document["meta"]["images"]

                            documents.append({"content": query, "meta": meta_data})
                        else:
                            meta_data = {
                                "name": path.name,
                            }
                            # Add image list parsed from docx into meta
                            if document["meta"] is not None and "images" in document["meta"]:
                                meta_data["images"] = document["meta"]["images"]
                            documents.append({"content": para, "meta": meta_data})
                else:
                    documents.append(
                        {"content": text, "meta": document["meta"] if "meta" in document else {"name": path.name}}
                    )
    return documents