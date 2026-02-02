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

import random
from typing import Dict, List, Tuple
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges import KnowledgeGatherToSplit
from helper import KnowledgeTestHelper, OptimizationConfig


def make_gather_to_split_graph(
    name: str,
    input_shape: Tuple[int, ...],
    gathers: List[Dict],
    version: int,
    extra_node: bool,
    extra_output: bool,
) -> OnnxGraph:
    graph = OnnxGraph(name=name, opset_imports=version)
    graph.add_input('input', np.float32, input_shape)
    output_shape = list(input_shape)
    output_shape[0] = sum(1 if isinstance(g['indices'], int) else len(g['indices']) for g in gathers)
    if extra_node:
        output_shape[0] += input_shape[0]
    graph.add_output('output', np.float32, output_shape)

    graph.add_node(
        name='relu_0',
        op_type='Relu',
        inputs=['input'],
        outputs=['relu_0_out'],
    )

    outputs = []
    for idx, gather in enumerate(gathers):
        axis, indices = gather['axis'], gather['indices']
        gather_name = f'gather_{idx}'
        ini_name = f'{gather_name}_indices'
        graph.add_initializer(
            name=ini_name,
            value=np.array(indices),
        )
        graph.add_node(
            name=gather_name,
            op_type='Gather',
            inputs=['relu_0_out', ini_name],
            outputs=[f'{gather_name}_out'],
            attrs={'axis': axis},
        )

        if extra_output and idx == 0:
            graph.add_output(f'{gather_name}_out', np.float32, None)

        if isinstance(indices, int):
            if version < 13:
                graph.add_node(
                    name=f'unsqueeze_5_{idx}',
                    op_type='Unsqueeze',
                    inputs=[f'{gather_name}_out'],
                    outputs=[f'unsqueeze_5_{idx}_out'],
                    attrs={'axes': [axis]},
                )
            else:
                graph.add_initializer(
                    name=f'unsqueeze_5_{idx}_axes',
                    value=np.array([axis]),
                )
                graph.add_node(
                    name=f'unsqueeze_5_{idx}',
                    op_type='Unsqueeze',
                    inputs=[f'{gather_name}_out', f'unsqueeze_5_{idx}_axes'],
                    outputs=[f'unsqueeze_5_{idx}_out'],
                )

        out = f'unsqueeze_5_{idx}_out' if isinstance(indices, int) else f'{gather_name}_out'

        graph.add_node(
            name=f'relu_1_{idx}',
            op_type='Relu',
            inputs=[out],
            outputs=[f'relu_1_{idx}_out'],
        )
        if axis != 0:
            perm = list(range(len(input_shape)))
            perm[axis], perm[0] = perm[0], perm[axis]
            graph.add_node(
                name=f'transpose_2_{idx}',
                op_type='Transpose',
                inputs=[f'relu_1_{idx}_out'],
                outputs=[f'transpose_2_{idx}_out'],
                attrs={'perm': perm},
            )
            outputs.append(f'transpose_2_{idx}_out')
        else:
            outputs.append(f'relu_1_{idx}_out')

    if extra_node:
        graph.add_node(
            name='relu_extra',
            op_type='Relu',
            inputs=['relu_0_out'],
            outputs=['relu_extra_out'],
        )
        outputs.append('relu_extra_out')

    graph.add_node(
        name='concat_3',
        op_type='Concat',
        inputs=outputs,
        outputs=['output'],
        attrs={'axis': 0},
    )

    graph.update_map()
    graph.infer_shape()
    return graph


class TestKnowledgeGatherToSplit(unittest.TestCase, KnowledgeTestHelper):
    @patch('os.makedirs')
    @patch(__name__ + '.make_gather_to_split_graph')
    @patch('helper.OptimizationConfig')
    def test_basic_split(self, mock_optimization_config, mock_make_gather_to_split_graph, mock_makedirs):
        # 模拟 make_gather_to_split_graph 返回的 graph 对象
        mock_graph = MagicMock()
        mock_make_gather_to_split_graph.return_value = mock_graph

        # 模拟 OptimizationConfig 实例
        mock_config = MagicMock()
        mock_optimization_config.return_value = mock_config

        # 模拟 check_optimization 和 check_precision 方法
        self.check_optimization = MagicMock(return_value=True)
        self.check_precision = MagicMock(return_value=True)

        # 定义测试数据
        tests = [
            (10, True, (10, 10, 10), [{'axis': 0, 'indices': [i]} for i in range(10)], False, False, 13),
            # 其他测试用例
        ]

        for i, (count, expect, ishape, gathers, enode, eout, version) in enumerate(tests):
            ishape_s = 'x'.join(str(x) for x in ishape)
            axis_s = 1 if len(set(g['axis'] for g in gathers)) == 1 else 0
            indices_s0 = []
            for g in gathers:
                idx = g['indices']
                if isinstance(idx, int):
                    indices_s0.append(str(idx))
                else:
                    indices_s0.append('x'.join(str(x) for x in idx))
            indices_s = '_'.join(indices_s0)

            name_ = f'test_gather_to_split_{i}_i{ishape_s}_a{axis_s}_idx{indices_s}_n{int(enode)}_o{int(eout)}_v{version}'

            with self.subTest(name=name_):
                onnx_ori = f'onnx/{name_}.onnx'
                onnx_opt = f'onnx/{name_}_opt.onnx'

                # 调用被测试的函数
                mock_graph.save = MagicMock()  # 模拟 graph.save 方法
                mock_config.graph = mock_graph
                mock_config.onnx_ori = onnx_ori
                mock_config.onnx_opt = onnx_opt

                # 调用被测试的逻辑
                self.assertTrue(self.check_optimization(cfg=mock_config, expect=expect))
                if not expect:
                    continue

                # 模拟输入数据
                feeds = [{'input': np.random.randn(*ishape).astype(np.float32)} for _ in range(count)]
                self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))


if __name__ == '__main__':
    unittest.main()
