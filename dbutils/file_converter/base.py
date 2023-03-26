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

from typing import Any, Callable, Optional, Dict, List, Tuple, Optional

from abc import abstractmethod
from pathlib import Path
import langdetect

from copy import deepcopy
from abc import abstractmethod
import inspect
import logging

from .schema import Document

logger = logging.getLogger(__name__)

class BaseComponent:
    """
    A base class for implementing nodes in a Pipeline.
    """

    outgoing_edges: int
    subclasses: dict = {}
    pipeline_config: dict = {}
    name: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        """
        Automatically keeps track of all available subclasses.
        Enables generic load() for all specific component implementations.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def get_subclass(cls, component_type: str):
        if component_type not in cls.subclasses.keys():
            raise Exception(f"pipelines component with the name '{component_type}' does not exist.")
        subclass = cls.subclasses[component_type]
        return subclass

    @classmethod
    def load_from_args(cls, component_type: str, **kwargs):
        """
        Load a component instance of the given type using the kwargs.

        :param component_type: name of the component class to load.
        :param kwargs: parameters to pass to the __init__() for the component.
        """
        subclass = cls.get_subclass(component_type)
        instance = subclass(**kwargs)
        return instance

    @classmethod
    def load_from_pipeline_config(cls, pipeline_config: dict, component_name: str):
        """
        Load an individual component from a YAML config for Pipelines.

        :param pipeline_config: the Pipelines YAML config parsed as a dict.
        :param component_name: the name of the component to load.
        """
        if pipeline_config:
            all_component_configs = pipeline_config["components"]
            all_component_names = [comp["name"] for comp in all_component_configs]
            component_config = next(comp for comp in all_component_configs if comp["name"] == component_name)
            component_params = component_config["params"]

            for key, value in component_params.items():
                if value in all_component_names:  # check if the param value is a reference to another component
                    component_params[key] = cls.load_from_pipeline_config(pipeline_config, value)

            component_instance = cls.load_from_args(component_config["type"], **component_params)
        else:
            component_instance = cls.load_from_args(component_name)
        return component_instance

    @abstractmethod
    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
    ) -> Tuple[Dict, str]:
        """
        Method that will be executed when the node in the graph is called.

        The argument that are passed can vary between different types of nodes
        (e.g. retriever nodes expect different args than a reader node)


        See an example for an implementation in pipelines/reader/base/BaseReader.py
        :return:
        """
        pass

    def _dispatch_run(self, **kwargs) -> Tuple[Dict, str]:
        """
        The Pipelines call this method which in turn executes the run() method of Component.

        It takes care of the following:
          - inspect run() signature to validate if all necessary arguments are available
          - pop `debug` and sets them on the instance to control debug output
          - call run() with the corresponding arguments and gather output
          - collate `_debug` information if present
          - merge component output with the preceding output and pass it on to the subsequent Component in the Pipeline
        """
        return self._dispatch_run_general(self.run, **kwargs)

    def _dispatch_run_batch(self, **kwargs):
        """
        The Pipelines call this method when run_batch() is executed. This method in turn executes the
        _dispatch_run_general() method with the correct run method.
        """
        return self._dispatch_run_general(self.run_batch, **kwargs)

    def _dispatch_run_general(self, run_method: Callable, **kwargs):
        """
        This method takes care of the following:
          - inspect run_method's signature to validate if all necessary arguments are available
          - pop `debug` and sets them on the instance to control debug output
          - call run_method with the corresponding arguments and gather output
          - collate `_debug` information if present
          - merge component output with the preceding output and pass it on to the subsequent Component in the Pipeline
        """
        arguments = deepcopy(kwargs)
        params = arguments.get("params") or {}

        run_signature_args = inspect.signature(run_method).parameters.keys()

        run_params: Dict[str, Any] = {}
        for key, value in params.items():
            if key == self.name:  # targeted params for this node
                if isinstance(value, dict):
                    # Extract debug attributes
                    if "debug" in value.keys():
                        self.debug = value.pop("debug")

                    for _k, _v in value.items():
                        if _k not in run_signature_args:
                            raise Exception(f"Invalid parameter '{_k}' for the node '{self.name}'.")

                run_params.update(**value)
            elif key in run_signature_args:  # global params
                run_params[key] = value

        run_inputs = {}
        for key, value in arguments.items():
            if key in run_signature_args:
                run_inputs[key] = value

        output, stream = run_method(**run_inputs, **run_params)

        # Collect debug information
        debug_info = {}
        if getattr(self, "debug", None):
            # Include input
            debug_info["input"] = {**run_inputs, **run_params}
            debug_info["input"]["debug"] = self.debug
            # Include output, exclude _debug to avoid recursion
            filtered_output = {key: value for key, value in output.items() if key != "_debug"}
            debug_info["output"] = filtered_output
        # Include custom debug info
        custom_debug = output.get("_debug", {})
        if custom_debug:
            debug_info["runtime"] = custom_debug

        # append _debug information from nodes
        all_debug = arguments.get("_debug", {})
        if debug_info:
            all_debug[self.name] = debug_info
        if all_debug:
            output["_debug"] = all_debug

        # add "extra" args that were not used by the node, but not the 'inputs' value
        for k, v in arguments.items():
            if k not in output.keys() and k != "inputs":
                output[k] = v

        output["params"] = params
        return output, stream

    def set_config(self, **kwargs):
        """
        Save the init parameters of a component that later can be used with exporting
        YAML configuration of a Pipeline.

        :param kwargs: all parameters passed to the __init__() of the Component.
        """
        if not self.pipeline_config:
            self.pipeline_config = {"params": {}, "type": type(self).__name__}
            for k, v in kwargs.items():
                if isinstance(v, BaseComponent):
                    self.pipeline_config["params"][k] = v.pipeline_config
                elif v is not None:
                    self.pipeline_config["params"][k] = v


class BaseConverter(BaseComponent):
    """
    Base class for implementing file converts to transform input documents to text format for ingestion in DocumentStore.
    """

    outgoing_edges = 1

    def __init__(
        self,
        remove_numeric_tables: bool = False,
        valid_languages: Optional[List[str]] = None,
    ):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages)

        self.remove_numeric_tables = remove_numeric_tables
        self.valid_languages = valid_languages

    @abstractmethod
    def convert(
        self,
        file_path: Path,
        meta: Optional[Dict[str, str]],
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "utf-8",
    ) -> List[Dict[str, Any]]:
        """
        Convert a file to a dictionary containing the text and any associated meta data.

        File converters may extract file meta like name or size. In addition to it, user
        supplied meta data like author, url, external IDs can be supplied as a dictionary.

        :param file_path: path of the file to convert
        :param meta: dictionary of meta data key-value pairs to append in the returned document.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        :param encoding: Select the file encoding (default is `utf-8`)
        """
        pass

    def validate_language(self, text: str, valid_languages: Optional[List[str]] = None) -> bool:
        """
        Validate if the language of the text is one of valid languages.
        """
        if valid_languages is None:
            valid_languages = self.valid_languages

        if not valid_languages:
            return True

        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = None

        return lang in valid_languages

    def run(  # type: ignore
        self,
        file_paths,  # type: ignore
        meta = None,  # type: ignore
        remove_numeric_tables: Optional[bool] = None,  # type: ignore
        valid_languages: Optional[List[str]] = None,  # type: ignore
    ):

        if isinstance(file_paths, Path):
            file_paths = [file_paths]

        if meta is None or isinstance(meta, dict):
            meta = [meta] * len(file_paths)  # type: ignore

        documents: list = []
        for file_path, file_meta in zip(file_paths, meta):
            for doc in self.convert(
                file_path=file_path,
                meta=file_meta,
                remove_numeric_tables=remove_numeric_tables,
                valid_languages=valid_languages,
            ):
                documents.append(doc)

        result = {"documents": documents}
        return result, "output_1"
