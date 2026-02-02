# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import stat
import json
import queue
from collections import Counter

from msit_llm.common.log import logger
from msit_llm.common.utils import load_file_to_read_common_check
from components.utils.file_open_check import ms_open
from components.utils.constants import TENSOR_MAX_SIZE
from components.utils.util import recursion_depth_decorator


FILE_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
MODULE_ID_NOT_AVAILABLE = -1
MIN_LAYER_NUMBER = 10


class TreeNode:
    def __init__(self, node_name: str, op_type: str, op_param=None, level=0, order=0, tensor_path="", show_order=0):
        self.node_name = node_name
        self.op_type = op_type
        self.op_param = op_param
        self.level = level
        self.order = order
        self.children = []
        self.tensor_path = tensor_path
        self.show_order = show_order

    def __repr__(self):
        return "{} [{}] [{}] ({})".format(
            self.node_name, self.tensor_path, self.op_type, ",".join((x.node_name for x in self.children))
        )

    def add_child(self, node):
        self.children.append(node)

    def sort_children(self):
        self.children.sort(key=lambda x: x.order)
        reorder = 0
        for sub_node in self.children:
            if sub_node.order != MODULE_ID_NOT_AVAILABLE:
                sub_node.order = reorder
                reorder = reorder + 1
            sub_node.sort_children()

    def get_next_sibling_node(self, node):
        """
        search next sibling node of the node
        """
        next_sibling_node = None
        node_q = queue.Queue()
        node_q.put(self)

        while not node_q.empty():
            parent_node = node_q.get()
            if node in parent_node.children:
                node_index = parent_node.children.index(node)
                if node_index + 1 < len(parent_node.children):
                    next_sibling_node = parent_node.children[node_index + 1]
                    break
            else:
                for child_node in parent_node.children:
                    node_q.put(child_node)

        return next_sibling_node

    def get_layer_node(self, layer_type: str):
        all_layer_nodes = []

        def run(node, layer_type, layer_nodes):
            for child_node in node.children:
                if child_node.op_type == layer_type:
                    layer_nodes.append(child_node)
                else:
                    run(child_node, layer_type, layer_nodes)

        run(self, layer_type, all_layer_nodes)
        return all_layer_nodes

    def get_layer_node_type(self):
        layer_node_type = ""

        def run(node):
            nonlocal layer_node_type
            if layer_node_type:
                return
            child_op_type = [child_node.op_type for child_node in node.children]
            if len(child_op_type) > MIN_LAYER_NUMBER:
                op_type_counts = Counter(child_op_type)
                most_count = op_type_counts.most_common(1)[0][1]
                if most_count > MIN_LAYER_NUMBER / 2:
                    most_op_type = op_type_counts.most_common(1)[0][0]
                    layer_node_type = most_op_type
            else:
                for child_node in node.children:
                    run(child_node)

        run(self)
        return layer_node_type

    def get_leaf_nodes(self):
        all_leaf_nodes = []

        def run(node, leaf_nodes):
            for child_node in node.children:
                if child_node.children:
                    run(child_node, leaf_nodes)
                else:
                    leaf_nodes.append(child_node)

        run(self, all_leaf_nodes)
        return all_leaf_nodes

    def get_all_nodes(self):
        all_nodes = [self]

        def run(node, children_nodes):
            for child_node in node.children:
                children_nodes.append(child_node)
                if child_node.children:
                    run(child_node, children_nodes)

        run(self, all_nodes)
        return all_nodes

    def get_next_level_nodes(self):
        next_level_nodes = []
        for child_node in self.children:
            next_level_nodes.extend(child_node.children)
        return next_level_nodes


class ModelTree:
    atb_show_order = 0

    def __init__(self):
        self.root_node = TreeNode("root", "root")

    @staticmethod
    def json_to_tree(json_path: str, tensor_path="") -> TreeNode:
        json_path = load_file_to_read_common_check(json_path)
        with ms_open(json_path, "r", max_size=TENSOR_MAX_SIZE) as file:
            node_dict = json.loads(file.read(), parse_constant=lambda x: None)

        def _dict_to_tree(node_dict, level, order, tensor_path):
            try:
                op_name = node_dict["name"]
            except Exception as e:
                logger.info(node_dict)
                raise e
            node_tensor_path = os.path.join(tensor_path, op_name)
            node = TreeNode(op_name, node_dict["type"], level, order, tensor_path=node_tensor_path)
            sub_level = level + 1
            sub_order = 0
            for child_dict in node_dict["children"]:
                child_node = _dict_to_tree(child_dict, sub_level, sub_order, tensor_path)
                node.add_child(child_node)
                sub_order = sub_order + 1
            return node

        return _dict_to_tree(node_dict, 0, 0, tensor_path)

    @staticmethod
    def atb_json_to_tree(json_path: str, tensor_path="", start_order=0) -> TreeNode:
        json_path = load_file_to_read_common_check(json_path)
        with ms_open(json_path, "r", max_size=TENSOR_MAX_SIZE) as file:
            node_dict = json.loads(file.read(), parse_constant=lambda x: None)

        def _atb_dict_to_tree(node_dict, level, order, tensor_path):
            ModelTree.atb_show_order = ModelTree.atb_show_order + 1
            if level == 0:
                node = TreeNode(
                    "root", node_dict["modelName"], tensor_path=tensor_path, show_order=ModelTree.atb_show_order
                )
            else:
                rel_path = str(order) + "_" + node_dict["opType"]
                tensor_path = os.path.join(tensor_path, rel_path)
                op_param = node_dict.get("param") if "param" in node_dict else None
                node = TreeNode(
                    node_dict["opName"],
                    node_dict["opType"],
                    op_param,
                    level,
                    order,
                    tensor_path,
                    show_order=ModelTree.atb_show_order,
                )

            if "nodes" in node_dict:
                if level == 0:
                    reorder = start_order  # 特殊处理，后面考虑优化
                else:
                    reorder = 0
                for child_dict in node_dict["nodes"]:
                    child_node = _atb_dict_to_tree(child_dict, level + 1, reorder, tensor_path)
                    reorder = reorder + 1
                    node.add_child(child_node)
            return node

        return _atb_dict_to_tree(node_dict, 0, 0, tensor_path)

    def create_tree(self, module, module_ids, json_path) -> None:
        self.root_node.op_type = str(type(module).__name__)
        self._create_sub_tree(module, self.root_node, module_ids)
        self.root_node.sort_children()
        _tree_to_json(self.root_node, json_path)

    def _create_sub_tree(self, module, father_node, module_ids):
        new_level = father_node.level + 1
        for sub_name, sub_module in module.named_children():
            new_name = father_node.node_name + "." + sub_name
            new_type = str(type(sub_module).__name__)
            new_order = module_ids.get(new_name, MODULE_ID_NOT_AVAILABLE)
            sub_node = TreeNode(new_name, new_type, new_level, new_order)
            father_node.add_child(sub_node)
            self._create_sub_tree(sub_module, sub_node, module_ids)


@recursion_depth_decorator('msit_llm.dump.torch_dump.topo._tree_to_dict')
def _tree_to_dict(node):
    return {
        "name": node.node_name,
        "type": node.op_type,
        "children": [_tree_to_dict(child) for child in node.children],
    }


def _tree_to_json(node, json_path):
    with ms_open(json_path, 'w') as file:
        json.dump(_tree_to_dict(node), file)
