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

from typing import Dict, Set, List

from model_evaluation.graph.onnx import OnnxGraph
from model_evaluation.common.enum import ONNXCheckerError


class OnnxChecker:
    def __init__(self, graph: OnnxGraph) -> None:
        self._graph = graph

    @staticmethod
    def _check_op_input(graph, err_map):
        nodes_outputs: Set[str] = set()
        for node in graph.node:
            nodes_outputs = nodes_outputs.union(node.output)

        model_inputs: Set[str] = set()
        for input_ in graph.input:
            model_inputs.add(input_.name)

        initializers: Set[str] = set()
        for initializer in graph.initializer:
            initializers.add(initializer.name)

        def is_valid(input_: str) -> bool:
            return input_ in nodes_outputs or input_ in initializers or input_ in model_inputs

        for node in graph.node:
            empty_input = [input_ for input_ in node.input if not is_valid(input_)]
            if len(empty_input) == 0:
                continue
            s = ','.join(empty_input)
            err_map.setdefault(node.name, []).append(f'input {s} is empty.')

    def check_ops(self) -> Dict[str, List[str]]:
        if self._graph is None:
            return {}

        graph = self._graph.graph

        err_map: Dict[str, List[str]] = {}
        self._check_op_validation_by_onnx_checker(graph, err_map)
        self._check_op_input(graph, err_map)
        return err_map

    def _check_op_validation_by_onnx_checker(self, graph, err_map):
        for node in graph.node:
            errcode, errinfo = self._graph.check_node(node)
            if errcode == ONNXCheckerError.UNREGISTERED_OP or errcode == ONNXCheckerError.SUCCESS:
                continue
            err_map.setdefault(node.name, []).append(errinfo)
