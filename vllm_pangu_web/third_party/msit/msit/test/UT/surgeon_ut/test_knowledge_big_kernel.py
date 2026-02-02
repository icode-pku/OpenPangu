# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import unittest

import numpy as np

from auto_optimizer import OnnxNode, OnnxInitializer, OnnxGraph
from auto_optimizer.pattern.knowledges import KnowledgeBigKernel
from auto_optimizer.pattern.knowledges.big_kernel.util import gen_normal_subgraph, MATMUL_W, ADD_B, RESHAPE_S,\
    TRANSPOSE, QK_MASK_ADD, END_ADD
from auto_optimizer.pattern.knowledges.big_kernel.util import K_TRANSPOSE2, START_ADD
from helper import KnowledgeTestHelper, OptimizationConfig


def gen_bert_attention():
    graph = gen_normal_subgraph(prefix="")
    graph.remove(name=K_TRANSPOSE2)

    start_add = graph.get_node(START_ADD, node_type=OnnxNode)
    start_add_b = graph.add_initializer(name=START_ADD + "_b", value=np.ones(([768])))

    graph.add_input(name="input1", dtype=np.dtype('float32'), shape=[16, 384, 768])
    start_add.inputs = ['input1', start_add_b.name]

    init_w = np.ones((768, 768)).astype("float32")
    init_bias = np.ones(768).astype("float32")
    init_shape = np.array([16, 384, 12, 64]).astype("int32")
    init_perm = [0, 2, 1, 3]
    branch_names = ["q_", "k_", "v_"]
    for branch_name in branch_names:
        mm_w = graph.get_node(branch_name + MATMUL_W, node_type=OnnxInitializer)
        mm_w.value = init_w

        add_b = graph.get_node(branch_name + ADD_B, node_type=OnnxInitializer)
        add_b.value = init_bias

        reshape_s = graph.get_node(branch_name + RESHAPE_S, node_type=OnnxInitializer)
        reshape_s.value = init_shape

        transpose = graph.get_node(branch_name + TRANSPOSE, node_type=OnnxNode)
        if branch_name == "k_":
            transpose.attrs = {"perm": [0, 2, 3, 1]}
        else:
            transpose.attrs = {"perm": init_perm}

    mm_w = graph.get_node(MATMUL_W, node_type=OnnxInitializer)
    mm_w.value = init_w
    add_b = graph.get_node(ADD_B, node_type=OnnxInitializer)
    add_b.value = init_bias

    mask_add = graph.get_node(QK_MASK_ADD, node_type=OnnxNode)
    graph.add_input(name='input_mask', dtype=np.dtype("float32"), shape=[16, 1, 1, 384])
    mask_add.inputs[1] = 'input_mask'

    end_add = graph.get_node(END_ADD, node_type=OnnxNode)
    end_add.inputs[1] = start_add.outputs[0]

    graph.update_map()
    return graph


def gen_gpt2_attention():
    graph = OnnxGraph("gpt2")
    input_x = graph.add_input(name="input_x", dtype="float32", shape=[64, 512, 768])
    start_add_b = graph.add_initializer(name="start_add_b", value=np.ones(768).astype("float32"))
    start_add = graph.add_node(name="start_add", op_type="Add", outputs=["start_add_output"],
                               inputs=[input_x.name, start_add_b.name])

    reshape1_s = graph.add_initializer(name="reshape1_s", value=np.array([32768, 768]).astype("float32"))
    reshape1 = graph.add_node(name="reshape1", op_type="Reshape", inputs=[start_add.outputs[0], reshape1_s.name],
                              outputs=["reshape1_output"])

    gemm1_w = graph.add_initializer(name="gemm1_w", value=np.ones([768, 2304]).astype("float32"))
    gemm1_b = graph.add_initializer(name="gemm1_b", value=np.ones([2304]).astype("float32"))
    gemm1 = graph.add_node(name="gemm1", op_type="Gemm",
                           attrs={"alpha": 1, "beta": 1},
                           inputs=[reshape1.outputs[0], gemm1_w.name, gemm1_b.name],
                           outputs=["gemm1_output"])

    reshape2_s = graph.add_initializer(name="reshape2_s", value=np.array([64, 512, 2304]).astype("float32"))
    reshape2 = graph.add_node(name="reshape2", op_type="Reshape", inputs=[gemm1.outputs[0], reshape2_s.name],
                              outputs=["reshape2_output"])

    split = graph.add_node(name="split", op_type="Split", attrs={"axis": 2}, inputs=[reshape2.outputs[0]],
                           outputs=["q", "k", "v"])
    q_transpose = gen_qkv_branch(graph, split.outputs[0], branch_name="q")
    k_transpose = gen_qkv_branch(graph, split.outputs[1], branch_name="k")
    v_transpose = gen_qkv_branch(graph, split.outputs[2], branch_name="v")

    qk_mm = graph.add_node(name='qk_mm', op_type="MatMul", inputs=[q_transpose.outputs[0], k_transpose.outputs[0]],
                           outputs=["qk_mm_output"])

    div_b = graph.add_initializer(name="div_b", value=np.array(8).astype("float32"))
    div = graph.add_node(name="div", op_type="Div", inputs=[qk_mm.outputs[0], div_b.name], outputs=["div_output"])

    sub_b = graph.add_initializer(name="sub_b", value=np.ones([1, 1, 512, 512]))
    sub = graph.add_node(name="sub", op_type="Sub", inputs=[div.outputs[0], sub_b.name], outputs=["sub_output"])

    softmax = graph.add_node(name="softmax", op_type="Softmax",
                             inputs=[sub.outputs[0]], outputs=["softmax_output"], attrs={"axis": -1})

    score_v_mm = graph.add_node(name="score_v_mm", op_type="MatMul",
                                inputs=[softmax.outputs[0], v_transpose.outputs[0]],
                                outputs=["score_v_mm_output"])

    transpose = graph.add_node(name="transpose", op_type="Transpose",
                               inputs=[score_v_mm.outputs[0]], outputs=["transpose_output"],
                               attrs={'perm': [0, 2, 1, 3]})

    reshape3_s = graph.add_initializer(name="reshape3_s", value=np.array([32768, 768]))
    reshape3 = graph.add_node(name="reshape3", op_type="Reshape",
                              inputs=[transpose.outputs[0], reshape3_s.name],
                              outputs=["reshape3_output"])

    gemm2_w = graph.add_initializer(name="gemm2_w", value=np.ones((768, 768)).astype("float32"))
    gemm2_b = graph.add_initializer(name="gemm2_b", value=np.ones(768).astype("float32"))
    gemm2 = graph.add_node(name="Gemm", op_type="Gemm",
                           attrs={"alpha": 1, "beta": 1},
                           inputs=[reshape3.outputs[0], gemm2_w.name, gemm2_b.name],
                           outputs=["gemm2_output"])

    reshape4_s = graph.add_initializer(name="reshape4_s", value=np.array([64, 512, 768]))
    reshape4 = graph.add_node(name="reshap4", op_type="Reshape",
                              inputs=[gemm2.outputs[0], reshape4_s.name], outputs=["reshape4_output"])

    end_add = graph.add_node(name="end_add", op_type="Add",
                             inputs=[reshape4.outputs[0], start_add.outputs[0]], outputs=["output"])

    output = graph.add_output(name="output", dtype=np.dtype("float32"), shape=[64, 512, 768])

    graph.update_map()
    return graph


def gen_qkv_branch(graph: OnnxGraph, input_x, branch_name="q"):
    reshape_s = graph.add_initializer(name=branch_name + "_reshape_s", value=np.array([64, 512, 12, 64]))
    reshape = graph.add_node(name=branch_name + "_reshape", op_type="Reshape",
                             inputs=[input_x, reshape_s.name],
                             outputs=[branch_name + "_reshape_output"])

    if branch_name == "k":
        transpose = graph.add_node(name=branch_name + "_transpose",
                                   op_type="Transpose",
                                   inputs=[reshape.outputs[0]],
                                   outputs=[branch_name + "_transpose_output"],
                                   attrs={"perm": [0, 2, 3, 1]})
    else:
        transpose = graph.add_node(name=branch_name + "_transpose",
                                   op_type="Transpose",
                                   inputs=[reshape.outputs[0]],
                                   outputs=[branch_name + "_transpose_output"],
                                   attrs={"perm": [0, 2, 1, 3]})
    return transpose


class TestKnowledgeBigKernel(unittest.TestCase, KnowledgeTestHelper):
    def test_big_kernel_opt_when_gpt2_then_pass(self):
        graph = gen_gpt2_attention()
        knowledge = KnowledgeBigKernel(graph, "start_add", "end_add")
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=knowledge,
            onnx_ori="./test_ori.onnx",
            onnx_opt="./test_opt.onnx"
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=True))

    def tearDown(self):
        super().tearDown()
        for filename in os.listdir('.'):
            if filename.startswith('test_o'):
                os.remove(filename)


if __name__ == '__main__':
    unittest.main()
