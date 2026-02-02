# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd. All rights reserved.
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

import re
import tensorflow.compat.v1 as tf

from components.debug.common import logger

MAX_LETTERS_PER_LINE = 2000


class TensorFlowGraphBuilder:
    def __init__(self, description):
        self.description = description
        self.graph = tf.Graph()
        self.graph_name = None
        self.nodes = {}
        self.axes = {}
        self.output_nodes = []
        # 解析描述并构建图
        self.parse_description()

    @staticmethod
    def _compute_diff_axes(origin_shape, target_shape, start_index=0):
        diff_axes = []
        for idx, (origin, target) in enumerate(zip(origin_shape, target_shape)):
            if origin != target:
                diff_axes.append(start_index + idx)
        return diff_axes
    
    @staticmethod
    def _build_data_node(node_info):
        repeats = node_info["attributes"]["repeats"]
        dtype = node_info["attributes"]["dtype"]
        return tf.placeholder(dtype=dtype, shape=repeats, name=node_info["name"])

    def parse_description(self):
        """解析描述并构建 TensorFlow 图, 当前仅支持静态shape已知场景"""
        with self.graph.as_default():
            self.parse_axes()
            self.parse_nodes()

    def parse_axes(self):
        """解析轴信息"""
        axis_pattern = re.compile(r"z(\d+)\(\d+\) : (\d+)")
        for line in self.description.splitlines():
            match = axis_pattern.search(line)
            if match:
                axis_name = f"z{match.group(1)}"
                axis_size = int(match.group(2))
                self.axes[axis_name] = axis_size

    def parse_nodes(self):
        """解析节点信息"""
        # mix_trans_512/block.0_1/mul/mul_3059
        graph_pattern = re.compile(r'Graph\: (\w+)')
        node_pattern = re.compile(r" ([\w/]+): (\w+) \((\d+)\)")
        repeat_pattern = re.compile(r"\.y\.repeats = \{([\d\s\,]+)\}")
        stride_pattern = re.compile(r"\.y\.strides = \{([\d\s\,]+)\}")

        current_node = None
        for line in self.description.splitlines():
            # defend redos attack
            if len(line) > MAX_LETTERS_PER_LINE:
                logger.warning("Some lines in the current file exceed the set limit of 2000 characters, \
                               which may cause parsing issues.")
                continue
            # 解析ascgraph name
            graph_match = graph_pattern.search(line)
            if graph_match:
                graph_name = graph_match.group(1)
                self.graph_name = graph_name
                continue

            node_match = node_pattern.search(line)
            if node_match:
                node_name = node_match.group(1)
                node_type = node_match.group(2)
                node_id = int(node_match.group(3))
                current_node = {
                    "name": node_name,
                    "type": node_type,
                    "id": node_id,
                    "inputs": [],
                    "outputs": [],
                    "attributes": {}
                }
                self.nodes[node_name] = current_node
                continue

            if line.strip().startswith(".x =") and len(line.split("=")) >= 2:
                input_name = line.split("=")[1].strip()
                input_name = re.sub(r"\.[^.]+$", "", input_name)
                current_node["inputs"].append(input_name)
            elif line.strip().startswith(".x1 =") and len(line.split("=")) >= 2:
                input_name = line.split("=")[1].strip()
                input_name = re.sub(r"\.[^.]+$", "", input_name)
                current_node["inputs"].append(input_name)
            elif line.strip().startswith(".x2 =") and len(line.split("=")) >= 2:
                input_name = line.split("=")[1].strip()
                input_name = re.sub(r"\.[^.]+$", "", input_name)
                current_node["inputs"].append(input_name)

            # 匹配输出
            if line.strip().startswith(".y.dtype") and len(line.split("=")) >= 2:
                dtype = line.split("=")[1].strip()
                current_node["attributes"]["dtype"] = dtype
            if line.strip().startswith(".y.axis") and len(line.split("=")) >= 2:
                axes = line.split("=")[1].strip().strip("{}").split(", ")
                current_node["attributes"]["axes"] = axes
            if line.strip().startswith(".y.repeats"):
                repeats_match = repeat_pattern.search(line)
                if repeats_match:
                    repeats = list(map(int, filter(None, repeats_match.group(1).split(", "))))
                    if not repeats:
                        repeats = [1]  # 标量
                    current_node["attributes"]["repeats"] = repeats
            if line.strip().startswith(".y.strides"):
                strides_match = stride_pattern.search(line)
                if strides_match:
                    strides = list(map(int, filter(None, strides_match.group(1).split(", "))))
                    current_node["attributes"]["strides"] = strides
                logger.debug(f"Node: {current_node['name']}")
                logger.debug(f"  Type: {current_node['type']}")
                logger.debug(f"  ID: {current_node['id']}")
                logger.debug(f"  Inputs: {current_node['inputs']}")
                logger.debug(f"  Attributes:")
                for attr, value in current_node["attributes"].items():
                    logger.debug(f"    {attr}: {value}")
        # 构建 TensorFlow 节点
        self.build_tensorflow_nodes()

    def build_tensorflow_nodes(self):
        """根据解析的节点信息构建 TensorFlow 节点"""
        node_type_to_tf_op = {
            "Data": self._build_data_node,
            "Load": self._build_identity_node,
            "Store": self._build_identity_node,
            "Output": self._build_output_node,
            "Broadcast": self._build_broadcast_node,
            "Mul": self._build_mul_node,
            "Add": self._build_add_node,
            "Abs": self._build_abs_node,
            "Relu": self._build_relu_node,
            "Div": self._build_div_node,
            "Cast": self._build_cast_node,
            "Sign": self._build_sign_node,
            "Exp": self._build_exp_node,
            "ReduceMean": self._build_reduce_mean_node,
            "Rsqrt": self._build_rsqt_node,
            "Sigmoid": self._build_sigmoid_node,
            "Sum": self._build_sum_mode,
        }

        for _, node_info in self.nodes.items():
            node_type = node_info["type"]
            if node_type in node_type_to_tf_op:
                node_info["output"] = node_type_to_tf_op[node_type](node_info)
            else:
                raise ValueError(f"Unsupported node type: {node_type}")

    def get_nodes(self):
        return {name: node["output"] for name, node in self.nodes.items()}

    def get_output_nodes(self):
        return [self.nodes[node_name]["output"] for node_name in self.output_nodes]

    def list_placeholders(self):
        return [tensor for tensor in self.graph.as_graph_def().node if tensor.op == 'Placeholder']

    def _get_input_node(self, node_info, index=0):
        """安全获取输入节点"""
        if not node_info.get("inputs") or index >= len(node_info["inputs"]):
            raise ValueError(f"Invalid input index {index} for node {node_info['name']}")
        input_name = node_info["inputs"][index]
        if input_name not in self.nodes:
            raise ValueError(f"Input node {input_name} not found for node {node_info['name']}")
        return self.nodes[input_name]["output"]

    def _build_identity_node(self, node_info):
        input_node = self._get_input_node(node_info)
        return tf.identity(input_node, name=node_info["name"])

    def _build_output_node(self, node_info):
        input_node = self._get_input_node(node_info)
        self.output_nodes.append(node_info["name"])
        return tf.identity(input_node, name=node_info["name"])

    def _build_broadcast_node(self, node_info):
        input_node = self._get_input_node(node_info)
        repeats = node_info["attributes"].get("repeats")
        if not repeats:
            raise ValueError(f"Missing repeats attribute for broadcast node {node_info['name']}")
        return tf.broadcast_to(input_node, repeats, name=node_info["name"])

    def _build_mul_node(self, node_info):
        input_node1 = self._get_input_node(node_info, 0)
        input_node2 = self._get_input_node(node_info, 1)
        return tf.multiply(input_node1, input_node2, name=node_info["name"])

    def _build_add_node(self, node_info):
        input_node1 = self._get_input_node(node_info, 0)
        input_node2 = self._get_input_node(node_info, 1)
        return tf.add(input_node1, input_node2, name=node_info["name"])

    def _build_abs_node(self, node_info):
        input_node = self._get_input_node(node_info)
        return tf.math.abs(input_node, name=node_info["name"])

    def _build_relu_node(self, node_info):
        input_node = self._get_input_node(node_info)
        return tf.nn.relu(input_node, name=node_info["name"])

    def _build_div_node(self, node_info):
        input_node1 = self._get_input_node(node_info, 0)
        input_node2 = self._get_input_node(node_info, 1)
        return tf.math.divide(input_node1, input_node2, name=node_info["name"])

    def _build_cast_node(self, node_info):
        input_node = self._get_input_node(node_info)
        dtype = node_info["attributes"].get("dtype")
        if not dtype:
            raise ValueError(f"Missing dtype attribute for cast node {node_info['name']}")
        return tf.cast(input_node, dtype=dtype, name=node_info["name"])

    def _build_sign_node(self, node_info):
        input_node = self._get_input_node(node_info)
        return tf.sign(input_node, name=node_info["name"])

    def _build_exp_node(self, node_info):
        input_node = self._get_input_node(node_info)
        return tf.math.exp(input_node, name=node_info["name"])

    def _build_reduce_mean_node(self, node_info):
        input_node = self._get_input_node(node_info)
        axis = node_info["attributes"].get("axis", -1)  # 默认最后一维
        keepdims = node_info["attributes"].get("keepdims", True)
        return tf.reduce_mean(input_node, axis=axis, keepdims=keepdims, name=node_info["name"])

    def _build_rsqt_node(self, node_info):
        input_node = self._get_input_node(node_info)
        return tf.math.rsqrt(input_node, name=node_info["name"])

    def _build_sigmoid_node(self, node_info):
        input_node = self._get_input_node(node_info)
        return tf.nn.sigmoid(input_node, name=node_info["name"])

    def _build_sum_mode(self, node_info):
        input_node = self._get_input_node(node_info)
        input_shape = input_node.shape.as_list()
        output_shape = node_info["attributes"].get("repeats")
        if not output_shape:
            raise ValueError(f"Missing repeats attribute for sum node {node_info['name']}")
        axis = self._compute_diff_axes(input_shape, output_shape)
        return tf.reduce_sum(input_node, name=node_info["name"], axis=axis)


def sanitize_filename(node_name: str):
    """
    将节点名称转换为有效的文件名。
    """
    return node_name.replace("/", "_")


def convert_to_tf_graph(pyautofuse_graph):
    from autofuse import pyautofuse # source CANN之后才能正确导入
    """将 pyautofuse 构建的图转换为 TensorFlow 图"""
    description = pyautofuse.ascir.utils.debug_str(pyautofuse_graph.graph)
    logger.debug("===========================ASC_GRAPH_DESCRIPTION======================================")
    logger.debug(description)
    logger.debug("======================================================================================")
    graph_builder = TensorFlowGraphBuilder(description)
    return graph_builder
