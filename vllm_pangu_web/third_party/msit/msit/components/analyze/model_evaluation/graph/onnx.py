# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from copy import deepcopy

import onnx
from onnx.onnx_cpp2py_export.checker import ValidationError
from google.protobuf.message import DecodeError

from model_evaluation.common.enum import ONNXCheckerError
from components.utils.check.rule import Rule


class OnnxGraph:
    def __init__(self, graph) -> None:
        super().__init__()
        self._graph = graph

    @property
    def graph(self):
        return self._graph.graph

    @classmethod
    def load(cls, model_path: str) -> 'OnnxGraph':
        if not os.path.isfile(model_path):
            raise RuntimeError(f'model {model_path} is not file.')

        try:
            Rule.input_file().check(model_path, will_raise=True)
            graph = onnx.load_model(model_path)
        except DecodeError as e:
            raise RuntimeError(f'{e}') from e

        return cls(graph)

    def opset_version(self) -> int:
        if self._graph is None:
            return -1

        opset_import = self._graph.opset_import
        if len(opset_import) == 0:
            return -1
        return opset_import[0].version

    def check_node(self, node):
        check_ctx = onnx.checker.DEFAULT_CONTEXT
        ori_opset_imports = deepcopy(check_ctx.opset_imports)
        check_ctx.opset_imports = {'': self.opset_version()}

        try:
            onnx.checker.check_node(node, check_ctx)
        except ValidationError as e:
            if str(e).find("No Op registered") != -1:
                return ONNXCheckerError.UNREGISTERED_OP, str(e)
            return ONNXCheckerError.OTHERS, str(e)

        check_ctx.opset_imports = ori_opset_imports
        return ONNXCheckerError.SUCCESS, ""
