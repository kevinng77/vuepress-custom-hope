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

from .base import BaseConverter

from .import_utils import safe_import

# MarkdownConverter = safe_import(
#     "pipelines.nodes.file_converter.markdown", "MarkdownConverter", "preprocessing"
# )  # Has optional dependencies
ImageToTextConverter = safe_import(
    "pipelines.nodes.file_converter.image", "ImageToTextConverter", "ocr"
)  # Has optional dependencies
PDFToTextConverter = safe_import(
    "pipelines.nodes.file_converter.pdf", "PDFToTextConverter", "ocr"
)  # Has optional dependencies
PDFToTextOCRConverter = safe_import(
    "pipelines.nodes.file_converter.pdf", "PDFToTextOCRConverter", "ocr"
)  # Has optional dependencies

from .docx import DocxToTextConverter
from .txt import TextConverter
from .pdf import PDFToTextConverter
# from .image import ImageToTextConverter
from .markdown import MarkdownConverter