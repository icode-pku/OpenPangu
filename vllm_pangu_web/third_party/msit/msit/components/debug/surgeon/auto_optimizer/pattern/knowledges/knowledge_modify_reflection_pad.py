# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd. All rights reserved.
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

import numpy as np

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.pattern.pattern import MatchPattern
from auto_optimizer.pattern.pattern import MatchBase
from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase

INT_MIN = -9223372036854775807


class ReflectionPadOpMatch(MatchBase):
    def __int__(self):
        super().__init__()

    def match(self, node: BaseNode, graph: BaseGraph):
        if node is None:
            return False
        if node.op_type != 'Pad':
            return False
        if graph.opset_imports[0].version < 10:  # do not support opset_version < 10
            return False
        if node['mode'] == b'reflect':  # Exception 1: the 3rd param (axes) provided
            if len(node.inputs) > 2:
                return False
            if len(node.inputs) == 2:  # opset >= 10
                if graph[node.inputs[1]] is None:  # Exception 2: cannot get padding value (usually from Initilizer)
                    return False
                else:
                    return True
        return False


@KnowledgeFactory.register()
class KnowledgeModifyReflectionPad(KnowledgeBase):
    def __init__(self):
        super().__init__()
        self.reflection_pad_op_pattern = Pattern() \
            .add_node('reflection_operator', ['Pad'], [ReflectionPadOpMatch()]) \
            .set_node_loop('reflection_operator', MatchPattern.MATCH_ONCE) \
            .set_loop(MatchPattern.MATCH_ONCE)

        self._register_apply_funcs(self.reflection_pad_op_pattern, [self._modify_reflection_pad_apply])

    def _construct_slice_ops(self, graph: BaseGraph, padding: int, func_option=0):
        """Given padding value, construct a group of `Slice` ops with size of 4, which could act as either
        slice operator or flip operator with different parameters (func_option)

        Args:
            graph: BaseGraph
            padding: int, padding value used in `Pad` op
            func_option: int, 0 or 1, indicate the function of `Slice` op:
                              0: 'Slice operation'
                              1: 'Flip operation'
        Return:
            slice_op_list: List[OnnxNode], a 4-size list of `Slice` op

        """

        if func_option == 0:  # slice op params
            slice_params = [
                [1, 1 + padding, 2, 1], [-1 - padding, -1, 2, 1],
                [1, 1 + padding, 3, 1], [-1 - padding, -1, 3, 1]
            ]
        else:  # flip op params
            slice_params = [
                [-1, INT_MIN, 2, -1], [-1, INT_MIN, 2, -1],
                [-1, INT_MIN, 3, -1], [-1, INT_MIN, 3, -1]
            ]

        slice_op_list = []
        for i, param_values in enumerate(slice_params):
            slice_op = graph.add_node(f'slice_{i}_{func_option}', 'Slice', outputs=[f'slice_{i}_{func_option}_output'])
            for j, param_name in enumerate(['start_', 'end_', 'axes_', 'step_']):
                initilizer = graph.add_initializer(
                    name=param_name + str(i) + str(j) + str(func_option),
                    value=np.array([param_values[j]]).astype(np.int64)
                )
                slice_op.inputs.append(initilizer.name)
            slice_op_list.append(slice_op)

        return slice_op_list

    def _modify_reflection_pad_apply(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        """Replace the origin pad op for a newly organized subgraph comprised of `Slice` and `Concat`

        """

        if match_result is None or match_result.is_empty():
            return False

        for node_dict in match_result.node_dicts:
            # retrieve info from origin pad op
            reflection_pad = node_dict.get('reflection_operator')[0]

            padding_input_node = reflection_pad.inputs[1]
            padding = graph[padding_input_node].value[-1]

            prev_node = reflection_pad.inputs[0]
            next_node = reflection_pad.outputs[0]

            # 1. construct four `Slice` ops for Slice
            slice_op_list = self._construct_slice_ops(graph, padding)

            if padding == 1:
                # 2. construct two `Concat` op
                concat_op1 = graph.add_node(
                    'concat_1', 'Concat',
                    inputs=[slice_op_list[0].outputs[0], prev_node, slice_op_list[1].outputs[0]],
                    outputs=['concat_1_output'],
                    attrs={'axis': 2}
                )
                concat_op2 = graph.add_node(
                    'concat_2', 'Concat',
                    inputs=[slice_op_list[2].outputs[0], concat_op1.outputs[0], slice_op_list[3].outputs[0]],
                    outputs=[next_node],
                    attrs={'axis': 3}
                )
            else:  # padding > 1
                flip_op_list = self._construct_slice_ops(graph, padding, func_option=1)

                # construct two `Concat` ops
                concat_op1 = graph.add_node(
                    'concat_1', 'Concat',
                    inputs=[flip_op_list[0].outputs[0], prev_node, flip_op_list[1].outputs[0]],
                    outputs=['concat_1_output'],
                    attrs={'axis': 2}
                )
                concat_op2 = graph.add_node(
                    'concat_2', 'Concat',
                    inputs=[flip_op_list[2].outputs[0], concat_op1.outputs[0], flip_op_list[3].outputs[0]],
                    outputs=[next_node],
                    attrs={'axis': 3}
                )

                # 3. link components
                slice_op_list[0].inputs.insert(0, prev_node)
                slice_op_list[1].inputs.insert(0, prev_node)
                slice_op_list[2].inputs.insert(0, concat_op1.outputs[0])
                slice_op_list[3].inputs.insert(0, concat_op1.outputs[0])

                # link `Flip` nodes
                for i in range(4):
                    flip_op_list[i].inputs.insert(0, slice_op_list[i].outputs[0])

                graph.remove(reflection_pad.name)
            graph.update_map()

            return True
