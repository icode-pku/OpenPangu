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

from collections import deque
from typing import Optional

import numpy as np

from auto_optimizer.graph_refactor import OnnxGraph, OnnxNode, OnnxInitializer
from auto_optimizer.pattern.knowledges.big_kernel.util import get_k_2nd_perm
from components.debug.common import logger
from auto_optimizer.pattern.knowledges.big_kernel.util import (
    MATMUL_W,
    RESHAPE_S,
    TRANSPOSE_PERM,
    K_TRANSPOSE_PERM2,
    MUL_B,
    ADD_B,
    CONVERT_3DIMS_TO_4DIMS,
    QK_MASK_ADD,
    QK_MASK_ADD_B,
)


class AttentionParser:
    def __init__(self, graph: OnnxGraph, start_node=None, end_node=None, softmax=None):
        self._graph = graph
        self._start_node = start_node
        self._end_node = end_node
        self._softmax = softmax
        self._params = {}
        self._branch_nodes = {}
        self._mask_add = None
        self._qk_mm = None
        self._score_v_mm = None

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph

    @property
    def start_node(self):
        return self._start_node

    @start_node.setter
    def start_node(self, node: OnnxNode):
        self._start_node = node

    @property
    def end_node(self):
        return self._end_node

    @end_node.setter
    def end_node(self, node: OnnxNode):
        self._end_node = node

    @property
    def softmax(self):
        return self._softmax

    @softmax.setter
    def softmax(self, node: OnnxNode):
        self._softmax = node

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    @property
    def branch_nodes(self):
        return self._branch_nodes

    @branch_nodes.setter
    def branch_nodes(self, value):
        self.branch_nodes = value

    def down_top_search(
        self, start_node: OnnxNode, goal_op_type: str, end_node=None, input_idx=0
    ) -> Optional[OnnxNode]:
        visited = set()
        if input_idx + 1 > len(start_node.inputs):
            return None
        goal_node = None
        pre_node = self.graph.get_prev_node(start_node.inputs[input_idx])
        if not pre_node:
            return None
        stack = [pre_node]
        while stack:
            node = stack.pop()
            visited.add(node)
            if node.op_type == goal_op_type:
                goal_node = node
                break
            if node == end_node:
                break
            for input_x in node.inputs:
                pre_node = self.graph.get_prev_node(input_x)
                if pre_node and pre_node not in visited:
                    stack.append(pre_node)
        return goal_node

    def top_down_search(self, start_node, goal_op_type, end_node=None) -> OnnxNode:
        goal_node = None
        visited = []
        queue = deque()
        queue.append(start_node)
        while queue:
            node = queue.popleft()
            if node == end_node:
                break
            if node != start_node and node.op_type == goal_op_type:
                goal_node = node
                break
            visited.append(node)
            for output_name in node.outputs:
                for next_node in self.graph.get_next_nodes(output_name):
                    if not next_node or next_node in visited:
                        continue
                    queue.append(next_node)
        return goal_node

    def parse_graph(self):
        self._qk_mm = self.down_top_search(self._softmax, goal_op_type="MatMul")
        self._parse_mask_add()
        self._parse_qkv()
        self._get_ori_shape()
        self._parse_last_linear_layer()

    def _parse_qkv(self):
        qkv_gemm = self.top_down_search(start_node=self.start_node, end_node=self._qk_mm, goal_op_type="Gemm")
        self._score_v_mm = self.top_down_search(self._softmax, goal_op_type="MatMul")
        if qkv_gemm:
            # 有些模型，如gpt2，qkv是合在一起用一个Gemm算子计算的，然后再split成3份
            self._concat_qkv_situation(qkv_gemm)
        else:
            # 有些模型，如bert，qkv分别用3个matmul计算的
            self._split_qkv_situation()

    def _parse_mask_add(self):
        """
        解析与mask相加的Add节点，这个节点可能是Add或者Sub，这个节点的另一个input可能是initializer，也可能是来自另一个分支
        """
        qk_mask_add = self.down_top_search(self.softmax, end_node=self._qk_mm, goal_op_type="Add")
        if qk_mask_add:
            qk_mask_add_b = self.graph.get_node(qk_mask_add.inputs[1], node_type=OnnxInitializer)
            if qk_mask_add_b:
                self._params.setdefault(QK_MASK_ADD_B, qk_mask_add_b.value)
            else:
                self._branch_nodes.setdefault(QK_MASK_ADD, qk_mask_add)
            self._mask_add = qk_mask_add
        else:
            qk_mask_sub = self.down_top_search(self._softmax, end_node=self._qk_mm, goal_op_type="Sub")
            if qk_mask_sub:
                qk_mask_sub_b = self.graph.get_node(qk_mask_sub.inputs[1], node_type=OnnxInitializer)
                if qk_mask_sub_b:
                    self._params.setdefault(QK_MASK_ADD_B, -1 * qk_mask_sub_b.value)
                else:
                    raise ValueError("The second input of node {} is not in initializer!".format(qk_mask_sub))
            else:
                raise ValueError("Add or Sub node is not found!")

    def _concat_qkv_situation(self, gemm):
        split_node = self.top_down_search(gemm, goal_op_type="Split")
        if not split_node:
            raise ValueError("There is not split node when concat qkv situation.")

        reshape = self.top_down_search(split_node, end_node=self._score_v_mm, goal_op_type="Reshape")
        reshape_s = self.graph.get_node(reshape.inputs[1], node_type=OnnxInitializer)

        gemm_w = self.graph.get_node(gemm.inputs[1], node_type=OnnxInitializer)
        gemm_b = self.graph.get_node(gemm.inputs[2], node_type=OnnxInitializer)
        w_shape = gemm_w.value.shape

        if w_shape[1] % 3 != 0:
            raise ValueError("{} can\'t be split into q, k, v.".format(gemm_w.name))

        split = int(w_shape[1] / 3)

        prefix_list = ["q_", "k_", "v_"]
        transpose_perm = []
        for i in range(3):
            start_idx = i * split
            end_idx = (i + 1) * split

            matmul_w = gemm_w.value[:, start_idx:end_idx]
            self._params.setdefault(prefix_list[i] + MATMUL_W, matmul_w)

            add_b = gemm_b.value[start_idx:end_idx]
            self._params.setdefault(prefix_list[i] + ADD_B, add_b)

            self._params.setdefault(prefix_list[i] + RESHAPE_S, reshape_s.value)

            if i == 0:
                # 通过计算q分支的transpose可以得到k和v的transpose的perm值
                q_transpose = self.down_top_search(
                    start_node=self._qk_mm, end_node=split_node, goal_op_type="Transpose"
                )
                transpose_perm = q_transpose.attrs.get("perm")

            self._params.setdefault(prefix_list[i] + TRANSPOSE_PERM, transpose_perm)

            if i == 2 and transpose_perm:
                k_transpose_perm2 = get_k_2nd_perm(transpose_perm)
                self._params.setdefault(K_TRANSPOSE_PERM2, k_transpose_perm2)

        div = self.top_down_search(start_node=self._qk_mm, end_node=self._score_v_mm, goal_op_type="Div")
        if div:
            div_b = self.graph.get_node(div.inputs[1], node_type=OnnxInitializer)
            try:
                mul_b = np.array(1 / div_b.value).astype(div_b.value.dtype)
            except ZeroDivisionError as err:
                logger.error("The value of {} is zero, please check your model.".format(div_b.name))
                raise err
        else:
            mul = self.top_down_search(self._qk_mm, end_node=self._score_v_mm, goal_op_type="Mul")
            if mul:
                mul_b = self.graph.get_node(mul.inputs[1], node_type=OnnxInitializer).value
            else:
                mul_b = np.array(1).astype(np.float32)

        self._params.setdefault(MUL_B, mul_b)

    def _split_qkv_situation(self):
        q_matmul = self._parse_qkv_branches()
        k_matmul = self._parse_qkv_branches(branch_name="k")
        v_matmul = self._parse_qkv_branches(branch_name="v")

        mul_b = np.array(1).astype(np.float32)
        all_mul_div_nodes = self._get_possible_mul_div(self._mask_add, q_matmul)
        for node in all_mul_div_nodes:
            if node.op_type == "Mul":
                bias = self.graph.get_node(node.inputs[1], node_type=OnnxInitializer).value
                mul_b = (mul_b * bias).astype(bias.dtype)

            if node.op_type == "Div":
                div_b = self.graph.get_node(node.inputs[1], node_type=OnnxInitializer)
                try:
                    mul_b = (mul_b / div_b.value).astype(div_b.value.dtype)
                except ZeroDivisionError as err:
                    logger.error("The value of {} is zero, please check your model.".format(div_b.name))
                    raise err

        self.params.setdefault(MUL_B, mul_b)

    def _parse_qkv_branches(self, branch_name="q"):
        """
        分别解析q、k、v3个分支上的matmul、add、reshape、transpose节点，并获取到参数和属性
        """
        prefix = branch_name + "_"
        if branch_name == "q":
            matmul = self.down_top_search(self._qk_mm, goal_op_type="MatMul")
            reshape = self.down_top_search(self._qk_mm, goal_op_type="Reshape")
        elif branch_name == "k":
            matmul = self.down_top_search(self._qk_mm, goal_op_type="MatMul", input_idx=1)
            reshape = self.down_top_search(self._qk_mm, goal_op_type="Reshape", input_idx=1)
        elif branch_name == "v":
            matmul = self.down_top_search(self._score_v_mm, goal_op_type="MatMul", input_idx=1)
            reshape = self.down_top_search(self._score_v_mm, goal_op_type="Reshape", input_idx=1)
        else:
            raise ValueError("Branch name must be one of qkv!")

        if not matmul:
            raise ValueError("Cannot find matmul node on branch {}".format(branch_name))
        matmul_w = self.graph.get_node(matmul.inputs[1], node_type=OnnxInitializer)
        self._params.setdefault(prefix + MATMUL_W, matmul_w.value)

        add = self.top_down_search(matmul, goal_op_type="Add")
        if not add:
            raise ValueError("Cannot find Add node after matmul on branch {}.".format(branch_name))

        for input_x in add.inputs:
            add_b = self.graph.get_node(input_x, node_type=OnnxInitializer)
            if add_b:
                self._params.setdefault(prefix + ADD_B, add_b.value)
                break

        if not reshape:
            raise ValueError("Cannot find reshape node on branch {}.".format(branch_name))
        reshape_s = self.graph.get_node(reshape.inputs[1], node_type=OnnxInitializer)
        if len(reshape_s.value) == 3:  # reshape必须得是4维，否则pattern匹配不上
            reshape_s_value = list(reshape_s.value)
            reshape_s_value.insert(0, 1)
            reshape_s_value = np.array(reshape_s_value)
            self._params.setdefault(CONVERT_3DIMS_TO_4DIMS, True)
        elif len(reshape_s.value) == 4:
            reshape_s_value = reshape_s.value
        else:
            raise ValueError("{} reshape node must be 3 or 4 dims".format(reshape))
        self._params.setdefault(prefix + RESHAPE_S, reshape_s_value)

        # 通过计算q分支的transpose可以得到k和v的transpose
        q_transpose = self.down_top_search(self._qk_mm, goal_op_type="Transpose")
        transpose_perm = q_transpose.attrs.get("perm")
        if self._params.get(CONVERT_3DIMS_TO_4DIMS):
            transpose_perm = [p + 1 for p in list(transpose_perm)]
            transpose_perm.insert(0, 0)
        self._params.setdefault(prefix + TRANSPOSE_PERM, transpose_perm)

        if branch_name == "k":
            k_transpose_perm2 = get_k_2nd_perm(transpose_perm)
            self._params.setdefault(K_TRANSPOSE_PERM2, k_transpose_perm2)

        return matmul

    def _get_possible_mul_div(self, mask_add, q_matmul):
        nodes = []
        mul1 = self.down_top_search(start_node=mask_add, goal_op_type="Mul", end_node=self._qk_mm)
        if mul1:
            nodes.append(mul1)

        div1 = self.down_top_search(start_node=mask_add, goal_op_type="Div", end_node=self._qk_mm)
        if div1:
            nodes.append(div1)

        mul2 = self.down_top_search(start_node=self._qk_mm, goal_op_type="Mul", end_node=q_matmul)
        if mul2:
            nodes.append(mul2)

        div2 = self.down_top_search(self._qk_mm, goal_op_type="Div", end_node=q_matmul)
        if div2:
            nodes.append(div2)

        return nodes

    def _get_ori_shape(self):
        # 获取attention的原始输入shape，用于在模型最后将feature map重新reshape成之前的shape
        ori_shape = None
        for idx, _ in enumerate(self.end_node.inputs):
            reshape = self.down_top_search(self.end_node, goal_op_type="Reshape", input_idx=idx)
            if reshape and reshape in self.graph.nodes:
                ori_shape = self.graph.get_node(reshape.inputs[1], node_type=OnnxInitializer)
                if ori_shape:
                    self._params.setdefault("ori_shape", ori_shape.value)

        if not ori_shape:
            raise ValueError("Cannot get origin shape of attention input.")

        # 各个attention和layernorm之间需要2维的shape计算，需要将attention的输出reshape成2维
        shape = ori_shape.value
        if len(shape) == 3:
            shape = np.array([shape[0] * shape[1], shape[2]])
        self._params.setdefault(RESHAPE_S, shape)

    def _parse_last_linear_layer(self):
        transpose = self.top_down_search(self._score_v_mm, goal_op_type="Transpose")
        perm = transpose.attrs.get("perm")
        if len(perm) == 3:
            perm = [p + 1 for p in perm]
            perm.insert(0, 0)
        self._params.setdefault("transpose_perm", perm)

        matmul = self.top_down_search(self._score_v_mm, goal_op_type="MatMul")
        if matmul:
            matmul_w = self.graph.get_node(matmul.inputs[1], node_type=OnnxInitializer)
            self._params.setdefault(MATMUL_W, matmul_w.value)
            add = self.top_down_search(matmul, goal_op_type="Add")
            for input_x in add.inputs:
                add_b = self.graph.get_node(input_x, node_type=OnnxInitializer)
                if add_b:
                    self._params.setdefault(ADD_B, add_b.value)
        else:
            gemm = self.top_down_search(self._score_v_mm, goal_op_type="Gemm")
            if gemm:
                weight, bias = self._parse_gemm_node(gemm)
                self._params.setdefault(MATMUL_W, weight)
                self._params.setdefault(ADD_B, bias)
            else:
                raise ValueError("There is no linear layer(gemm or matmul) at the end of the multi-head attention.")

    def _parse_gemm_node(self, gemm):
        """
        将attention中的Gemm算子转换成matmul算子，因此需要解析Gemm算子的weight和bias
        Gemm的计算公式如下：
            A' = transpose(A) if transA else A
            B' = transpose(B) if transB else B
            Y = alpha * A' * B' + beta * C
        所以，matmul的weight=alpha*B, bias=beta*C
        """
        attrs = gemm.attrs
        gemm_w = self.graph.get_node(gemm.inputs[1], node_type=OnnxInitializer)
        if attrs.get("transB"):  # transA的情况暂不考虑，需要在gemm（matmul）之前插入transpose算子，这样将破坏了标准pattern
            weight = attrs.get("alpha") * gemm_w.value.T
        else:
            weight = attrs.get("alpha") * gemm_w.value

        if len(gemm.inputs) == 3:
            gemm_b = self.graph.get_node(gemm.inputs[2], node_type=OnnxInitializer)
            bias = attrs.get("beta") * gemm_b.value
        else:  # Gemm没有偏置的情况
            w_shape = self._params.get(MATMUL_W).shape
            bias = np.zeros([w_shape[-1]]).astype(np.float32)
        return weight, bias
