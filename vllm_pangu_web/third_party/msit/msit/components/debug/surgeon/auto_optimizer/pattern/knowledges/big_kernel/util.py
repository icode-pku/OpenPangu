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

import numpy as np

from auto_optimizer.graph_refactor.onnx import OnnxGraph

START_ADD = "start_add"

Q_MATMUL = "q_matmul"
Q_MATMUL_W = "q_matmul_w"
Q_ADD = "q_add"
Q_ADD_B = "q_add_b"
Q_RESHAPE = "q_reshape"
Q_RESHAPE_S = "q_reshape_s"
Q_TRANSPOSE = "q_transpose"
Q_TRANSPOSE_PERM = "q_transpose_perm"

K_MATMUL = "k_matmul"
K_MATMUL_W = "k_matmul_w"
K_ADD = "k_add"
K_ADD_B = "k_add_b"
K_RESHAPE = "k_reshape"
K_RESHAPE_S = "k_reshape_s"
K_TRANSPOSE1 = "k_transpose1"
K_TRANSPOSE2 = "k_transpose2"
K_TRANSPOSE_PERM = "k_transpose_perm"
K_TRANSPOSE_PERM2 = "k_transpose2_perm"

V_MATMUL = "v_matmul"
V_MATMUL_W = "v_matmul_w"
V_ADD = "v_add"
V_ADD_B = "v_add_b"
V_RESHAPE = "v_reshape"
V_RESHAPE_S = "v_reshape_s"
V_TRANSPOSE = "v_transpose"
V_TRANSPOSE_PERM = "v_transpose_perm"

QK_MATMUL = "qk_matmul"
MUL = "mul"
MUL_B = "mul_b"
QK_MASK_ADD = "qk_mask_add"
QK_MASK_ADD_B = "qk_mask_add_b"
SOFTMAX = "softmax"

SCORE_V_MATMUL = "score_v_matmul"
TRANSPOSE = "transpose"
TRANSPOSE_PERM = "transpose_perm"
RESHAPE = "reshape"
RESHAPE_S = "reshape_s"
MATMUL = "matmul"
MATMUL_W = "matmul_w"
ADD = "add"
ADD_B = "add_b"

END_ADD = "end_add"
END_ADD_B = "end_add_b"

CONVERT_3DIMS_TO_4DIMS = "convert_3dims_to_4dims"


def get_k_2nd_perm(q_perm):
    """
    k有两个transpose，其中第一个transpose的值等于q_transpose_perm，
    第二个transpose的值等于q_transpose_perm的后两个元素交换，改变维度才能做矩阵乘。
    """
    dims = len(q_perm)
    k_transpose_perm2 = list(range(len(q_perm)))
    for axis in range(len(q_perm)):
        if dims - axis == 2:
            k_transpose_perm2[axis] = axis + 1
        if dims - axis == 1:
            k_transpose_perm2[axis] = axis - 1
    return k_transpose_perm2


def gen_normal_subgraph(prefix: str = "0."):
    subgraph = OnnxGraph(name="normal_pattern")
    init_value = np.zeros((1, 1))

    add_name = prefix + START_ADD
    add_output_name = add_name + "_output"
    add_node = subgraph.add_node(add_name, op_type="Add", outputs=[add_output_name])

    q_transpose = gen_qkv_block(subgraph, block_input=add_output_name, prefix=prefix)
    k_transpose = gen_qkv_block(subgraph, block_input=add_output_name, block_name="k", prefix=prefix)
    v_transpose = gen_qkv_block(subgraph, block_input=add_output_name, block_name="v", prefix=prefix)

    qk_mm_name = prefix + QK_MATMUL
    qk_mm = subgraph.add_node(
        name=qk_mm_name, op_type="MatMul",
        inputs=[q_transpose.outputs[0], k_transpose.outputs[0]],
        outputs=[qk_mm_name + "_output"]
    )

    mul_name = "bert_" + prefix + MUL  # attentionScore的pattern融合时需要识别mul算子名称以bert_开头
    mul_output_name = mul_name + "_output"
    mul_b = subgraph.add_initializer(name=prefix + MUL_B, value=init_value)
    mul = subgraph.add_node(name=mul_name, op_type="Mul",
                            inputs=[qk_mm.outputs[0], mul_b.name], outputs=[mul_output_name])

    qk_mask_add_name = prefix + QK_MASK_ADD
    qk_mask_add_b = subgraph.add_initializer(name=prefix + QK_MASK_ADD_B, value=init_value)
    qk_mask_add = subgraph.add_node(name=qk_mask_add_name, op_type="Add",
                                    inputs=[mul_output_name, qk_mask_add_b.name],
                                    outputs=[qk_mask_add_name + "_output"])

    softmax_name = prefix + SOFTMAX
    softmax_output_name = softmax_name + "_output"
    softmax = subgraph.add_node(name=softmax_name, attrs={"axis": -1}, op_type="Softmax",
                                inputs=[qk_mask_add.outputs[0]],
                                outputs=[softmax_output_name])

    score_v_mm_name = prefix + SCORE_V_MATMUL
    score_v_mm = subgraph.add_node(name=score_v_mm_name, op_type="MatMul",
                                   inputs=[softmax_output_name, v_transpose.outputs[0]],
                                   outputs=[score_v_mm_name + "_output"])

    transpose_name = prefix + TRANSPOSE
    transpose = subgraph.add_node(name=transpose_name, op_type="Transpose",
                                  attrs={"perm": []},
                                  inputs=[score_v_mm.outputs[0]],
                                  outputs=[transpose_name + "_output"])

    reshape_name = prefix + RESHAPE
    reshape_s = subgraph.add_initializer(name=prefix + RESHAPE_S, value=init_value)
    reshape = subgraph.add_node(name=reshape_name, op_type="Reshape",
                                inputs=[transpose.outputs[0], reshape_s.name],
                                outputs=[reshape_name + "_output"])

    mm_name = prefix + MATMUL
    mm_w = subgraph.add_initializer(name=prefix + MATMUL_W, value=init_value)
    mm = subgraph.add_node(name=mm_name, op_type="MatMul",
                           inputs=[reshape.outputs[0], mm_w.name], outputs=[mm_name + "_output"])

    add_name = prefix + ADD
    add_output = add_name + "_output"
    add_b = subgraph.add_initializer(name=prefix + ADD_B, value=init_value)
    add = subgraph.add_node(name=add_name, op_type="Add",
                            inputs=[mm.outputs[0], add_b.name], outputs=[add_output])

    end_add_name = prefix + END_ADD
    end_add_b = subgraph.add_initializer(name=prefix + END_ADD_B, value=init_value)
    end_add = subgraph.add_node(name=end_add_name, op_type="Add",
                                inputs=[add_output, end_add_b.name],
                                outputs=[end_add_name + "_output"]
                                )
    return subgraph


def gen_qkv_block(subgraph: OnnxGraph, block_input, block_name="q", prefix=""):
    init_value = np.zeros((1, 1))
    init_perm = [1, 1, 1, 1]
    prefix = prefix + block_name + "_"
    if block_name not in ["q", "k", "v"]:
        raise ValueError("Invalid block name {}".format(block_name))

    mm_name = prefix + MATMUL
    mm_w = subgraph.add_initializer(name=prefix + MATMUL_W, value=init_value)
    mm_node = subgraph.add_node(name=mm_name, op_type="MatMul", inputs=[block_input, mm_w.name],
                                outputs=[mm_name + "_output"])

    add_name = prefix + ADD
    add_b = subgraph.add_initializer(name=prefix + ADD_B, value=init_value)
    add_node = subgraph.add_node(add_name, op_type="Add", inputs=[mm_node.outputs[0], add_b.name],
                                 outputs=[add_name + "_output"])

    reshape_name = prefix + RESHAPE
    reshape_s = subgraph.add_initializer(name=reshape_name + "_s", value=init_value)
    reshape_node = subgraph.add_node(name=reshape_name, op_type="Reshape",
                                     inputs=[add_node.outputs[0], reshape_s.name],
                                     outputs=[reshape_name + "_output"]
                                     )

    transpose_name = prefix + TRANSPOSE
    transpose = subgraph.add_node(name=transpose_name, op_type="Transpose", inputs=[reshape_node.outputs[0]],
                                  outputs=[transpose_name + "_output"], attrs={"perm": init_perm})
    end_node = transpose

    if block_name == "k":
        transpose2_name = prefix + "transpose2"
        transpose2_output = transpose2_name + "_output"
        transpose2 = subgraph.add_node(name=transpose2_name, op_type="Transpose", inputs=[transpose.outputs[0]],
                                       outputs=[transpose2_output], attrs={"perm": init_perm}
                                       )
        end_node = transpose2
    return end_node
