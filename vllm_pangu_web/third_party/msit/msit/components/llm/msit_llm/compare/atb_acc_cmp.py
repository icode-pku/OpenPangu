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

import glob
import json
import os
import re

from tqdm import tqdm
from msit_llm.common.log import logger
from msit_llm.common.utils import load_file_to_read_common_check
from msit_llm.compare.cmp_utils import BasicDataInfo, fill_row_data, save_compare_result_to_csv, compare_data, read_data
from msit_llm.compare.cmp_op_match import MatchLocation
from msit_llm.compare.op_mapping import ATB_TORCH_BUILT_IN_OP_OUTPUT_MAPPING, ATB_TORCH_CUSTOM_OP_OUTPUT_MAPPING, \
    ATB_QUANT_FLOAT_NODE_MAPPING
from msit_llm.dump.torch_dump.topo import ModelTree, TreeNode
from components.utils.file_open_check import ms_open
from components.utils.constants import TENSOR_MAX_SIZE


def compare_file(golden_path, my_path):
    golden_data = read_data(golden_path)
    my_data = read_data(my_path)
    res = compare_data(golden_data, my_data)
    logger.info("Compared results: %s", res)


def traverse_tree(node: dict, path, traverse_type='torch', node_id=''):
    def enumerate_children(children, path, traverse_type='torch_model', node_id=''):
        res = []
        for idx, children_node in enumerate(children):
            if node_id != '':
                res.extend(traverse_tree(children_node, path, traverse_type, node_id + f'_{idx}'))
            else:
                res.extend(traverse_tree(children_node, path, traverse_type, str(idx)))
        return res

    res = []  # 用于保存遍历模型topo结构后得到的节点列表
    node['id'] = node_id
    if traverse_type == 'torch':
        node['golden_path'] = os.path.join(os.path.abspath(path), node['name'])
        res.append(node)
        if len(node['children']) > 0:
            res.extend(enumerate_children(node['children'], path, traverse_type, node_id))
    else:
        node['my_path'] = os.path.join(os.path.abspath(path), '_*/'.join(node_id.split('_')) + '_*', 'after')
        res.append(node)
        if 'nodes' in node.keys() and len(node['nodes']) > 0:
            res.extend(enumerate_children(node['nodes'], path, traverse_type, node_id))
    return res


def add_specific_path(golden_tensor, my_tensor, matched_path_pair):
    try:
        _golden_path = glob.glob(golden_tensor)[0]
        golden_out_path = get_paths(_golden_path, split_pattern='outtensor')
        _my_path = glob.glob(my_tensor)[0]
        my_out_path = get_paths(_my_path, split_pattern='outtensor')
        for _golden_tensor_path, _my_tensor_path in zip(golden_out_path, my_out_path):
            matched_path_pair.append({'golden': _golden_tensor_path, 'my': _my_tensor_path})
    except IndexError:
        msg = f"Cannot find path! golden: {golden_tensor}, my: {my_tensor}"
        logger.debug(msg)


def get_matched_path_pair(matches):
    matched_path_pair = []
    for match in matches:
        add_specific_path(match['golden']['my_path'], match['my']['my_path'], matched_path_pair)
    return matched_path_pair


def search_mapping_relationships(gathered_golden_data, gathered_my_data):
    matches = []
    for golden_item, my_item in zip(gathered_golden_data, gathered_my_data):
        if "opType" in golden_item and "opType" in my_item:
            matches.append({'golden': golden_item, 'my': my_item})

    return get_matched_path_pair(matches)


def search_float_quant_matches(golden_path, my_path, golden_topo_json_path, my_topo_json_path):
    matched_path_pair = []
    golden_tree = ModelTree.atb_json_to_tree(golden_topo_json_path)
    my_tree = ModelTree.atb_json_to_tree(my_topo_json_path)

    def get_subnode_by_name(target_node, target_name):
        for next_node in target_node.children:
            if next_node.node_name == target_name:
                return next_node
        return None

    def type_based_matches(my_node, golden_node):
        my_type_map = {}
        golden_type_map = {}
        my_legal_opname = {}
        for sub_node in my_node.children:
            match_type = sub_node.op_type
            if match_type == 'LinearOperation':
                match_type = 'LinearQuantOperation'
            if match_type not in my_type_map:
                my_type_map[match_type] = []
            my_type_map[match_type].append(sub_node.node_name)
        for sub_node in golden_node.children:
            if sub_node.op_type not in golden_type_map:
                golden_type_map[sub_node.op_type] = []
            golden_type_map[sub_node.op_type].append(sub_node.node_name)
        for key, value in my_type_map.items():
            if (key in golden_type_map) and (len(value) == len(golden_type_map[key])):
                for my_name, golden_name in zip(my_type_map.get(key, 'null'), golden_type_map.get(key, 'null')):
                    my_legal_opname[my_name] = golden_name
            elif key in ATB_QUANT_FLOAT_NODE_MAPPING and (ATB_QUANT_FLOAT_NODE_MAPPING[key] in golden_type_map) and \
                    (len(value) == len(golden_type_map[ATB_QUANT_FLOAT_NODE_MAPPING[key]])):
                for my_name, golden_name in zip(my_type_map.get(key, 'null'), 
                                                golden_type_map.get(ATB_QUANT_FLOAT_NODE_MAPPING[key], 'null')):
                    my_legal_opname[my_name] = golden_name
        for my_sub in my_node.children:
            if my_sub.node_name in my_legal_opname:
                my_level = '_*/'.join(my_sub.node_name.split("_", 1)[1].split('_')) + '_*'
                golden_level = '_*/'.join(my_legal_opname[my_sub.node_name].split("_", 1)[1].split('_')) + '_*'
                my_tensor = os.path.join(os.path.abspath(my_path), my_level, 'after')
                golden_tensor = os.path.join(os.path.abspath(golden_path), golden_level, 'after')
                add_specific_path(golden_tensor, my_tensor, matched_path_pair)
                type_based_matches(my_sub, get_subnode_by_name(golden_node, my_legal_opname[my_sub.node_name]))

    type_based_matches(my_tree, golden_tree)
    return matched_path_pair


def get_paths(path_dir, split_pattern):
    out_paths = [x for x in os.listdir(path_dir) if x.startswith('out')]
    out_paths.sort(key=lambda x: int(x.split(split_pattern)[-1].split('.')[0]))
    out_paths = [os.path.join(path_dir, x) for x in out_paths]
    return out_paths


def pair_built_in_op(g_nodes, m_nodes, op_mapping, my_root_node: TreeNode, callback=None):
    compared_result = []
    for atb_op_type, torch_op_type in op_mapping.items():
        atb_nodes = [m_node for m_node in m_nodes if m_node.op_type == atb_op_type]
        torch_nodes = [g_node for g_node in g_nodes if g_node.op_type == torch_op_type]
        if len(atb_nodes) != len(torch_nodes):
            msg = f"The number of {atb_op_type} node in atb is not equal to {torch_op_type} node in torch"
            logger.debug(msg)
            continue
        for atb_node, torch_node in zip(atb_nodes, torch_nodes):
            if atb_node.op_type == "LinearOperation" and not atb_node.op_param.get("hasBias"):
                next_sibling_node = my_root_node.get_next_sibling_node(atb_node)
                # 当有些算子如ParallelLinearBaseV2，是将w*x+b的操作拆分成两个算子，linear+add，而torch中使用一个算子Linear实现，
                # 因此add node的输出映射的是torch中Linear的输出
                if next_sibling_node and next_sibling_node.op_type == "ElewiseOperation" \
                        and next_sibling_node.op_param.get('elewiseType') == 8:
                    atb_node = next_sibling_node
            if callback is not None:
                callback(torch_node, 'output.pth', atb_node, 'outtensor0.bin')
                continue
            if callback is not None:
                callback(torch_node, 'output.pth', atb_node, 'outtensor0.bin')
                continue
            my_tensor_path = os.path.join(atb_node.tensor_path, "after", "outtensor0.bin")
            golden_tensor_path = os.path.join(torch_node.tensor_path, "output.pth")
            if os.path.exists(golden_tensor_path) and os.path.exists(my_tensor_path):
                data_info = BasicDataInfo(golden_tensor_path, my_tensor_path)
                row_data = fill_row_data(data_info)
                compared_result.append(row_data)
            else:
                msg = f"golden tensor path: {golden_tensor_path} or my_tensor_path: {my_tensor_path} is not exist."
                logger.debug(msg)
    return compared_result


def pair_custom_op(g_nodes, m_nodes, op_mapping, callback=None):
    compared_result = []

    op_mapping_flat = []
    for atb_op_type, torch_op_type in op_mapping.items():
        op_mapping_flat.extend([(atb_op_type, x) for x in torch_op_type])

    for atb_op_type, torch_op_type in op_mapping_flat:
        if '_' in atb_op_type:
            atb_op_type, atb_output = atb_op_type.split('_', 1)[0], atb_op_type.split('_', 1)[1]
            collect_atb_output = f"{atb_output}.bin"
        else:
            atb_output = "outtensor0"
            collect_atb_output = MatchLocation.ALL_OUTPUT
        if '_' in torch_op_type:
            torch_op_type, torch_output = torch_op_type.split('_', 1)[0], torch_op_type.split('_', 1)[1]
            collect_torch_output = f"{torch_output}.pth"
        else:
            torch_output = "outtensor0"
            collect_torch_output = MatchLocation.ALL_OUTPUT
        atb_nodes = [m_node for m_node in m_nodes if atb_op_type in m_node.op_type]
        torch_nodes = [g_node for g_node in g_nodes if torch_op_type in g_node.op_type]
        if len(atb_nodes) != len(torch_nodes):
            msg = f"The number of {atb_op_type} node in atb is not equal to {torch_op_type} node in torch"
            logger.debug(msg)
            continue
        for atb_node, torch_node in zip(atb_nodes, torch_nodes):
            if callback is not None:
                callback(torch_node, collect_torch_output, atb_node, collect_atb_output)
                continue
            my_tensor_path = os.path.join(atb_node.tensor_path, "after", f"{atb_output}.bin")
            golden_tensor_path = os.path.join(torch_node.tensor_path, f"{torch_output}.pth")
            if os.path.exists(golden_tensor_path) and os.path.exists(my_tensor_path):
                data_info = BasicDataInfo(golden_tensor_path, my_tensor_path)
                row_data = fill_row_data(data_info)
                compared_result.append(row_data)
            else:
                msg = f"golden tensor path: {golden_tensor_path} or my_tensor_path: {my_tensor_path} is not exist."
                logger.debug(msg)

    return compared_result


def cmp_torch_atb_model(data_info, output_path, mapping_dic):
    golden_json = data_info.get("golden_json")
    my_json = data_info.get("my_json")
    torch_tensor_path = data_info.get("torch_tensor_path")
    atb_tensor_path = data_info.get("atb_tensor_path")
    compared_result = []

    golden_root_node = ModelTree.json_to_tree(golden_json, torch_tensor_path)
    golden_layer_type = golden_root_node.get_layer_node_type()
    logger.info("golden_layer_type: %s", golden_layer_type)
    golden_layer_nodes = golden_root_node.get_layer_node(golden_layer_type)

    my_root_node = ModelTree.atb_json_to_tree(my_json, atb_tensor_path)
    my_layer_type = my_root_node.get_layer_node_type()
    logger.info("my_layer_type: %s", my_layer_type)
    my_layer_nodes = my_root_node.get_layer_node(my_layer_type)

    # 原生算子比对
    for golden_layer, my_layer in zip(golden_layer_nodes, my_layer_nodes):
        g_layer_leaf_nodes = golden_layer.get_leaf_nodes()
        m_layer_leaf_nodes = my_layer.get_leaf_nodes()
        compared_result.extend(pair_built_in_op(g_layer_leaf_nodes, m_layer_leaf_nodes,
                                                mapping_dic.get("ATB_TORCH_BUILT_IN_OP_OUTPUT_MAPPING"), my_root_node,
                                                atb_tensor_path, torch_tensor_path))

    # 自定义算子比对
    for golden_layer, my_layer in zip(golden_layer_nodes, my_layer_nodes):
        g_layer_all_nodes = golden_layer.get_all_nodes()
        m_layer_all_nodes = my_layer.get_all_nodes()
        compared_result.extend(pair_custom_op(g_layer_all_nodes, m_layer_all_nodes,
                                              mapping_dic.get("ATB_TORCH_CUSTOM_OP_OUTPUT_MAPPING")))

    return save_compare_result_to_csv(compared_result, output_path)


def get_only_dir_in_path(p_path):
    file_list = os.listdir(p_path)

    if len(file_list) != 1 or not os.path.isdir(os.path.join(p_path, file_list[0])):
        logger.warning(f"There must be only one directory in the {p_path}")
        return None
    
    return os.path.join(p_path, file_list[0])


def cmp_torch_atb_token(torch_tensor_path, atb_tensor_path, token_id):
    compare_result = []

    torch_layer_path = get_only_dir_in_path(torch_tensor_path)
    atb_layer_path = get_only_dir_in_path(atb_tensor_path)

    if atb_tensor_path is None or torch_tensor_path is None:
        return compare_result
    
    golden_tensor_path = os.path.join(torch_layer_path, "output.pth")
    my_tensor_path = os.path.join(atb_layer_path, "after", "outtensor0.bin")

    if os.path.exists(golden_tensor_path) and os.path.exists(my_tensor_path):
        data_info = BasicDataInfo(golden_tensor_path, my_tensor_path, token_id=token_id)
        row_data = fill_row_data(data_info)
        compare_result.append(row_data)

    return compare_result


def validate_json(json_obj):
    for key, value in json_obj.items():
        if not re.match(r"^[a-zA-Z0-9_]*$", key):
            return False
        if not isinstance(value, str) and not isinstance(value, list):
            return False
        if isinstance(value, str):
            if not re.match(r"^[a-zA-Z0-9_]*$", value):
                return False
        if isinstance(value, list):
            for v in value:
                if not re.match(r"^[a-zA-Z0-9_]*$", v):
                    return False
    return True


def load_mapping(mapping_file_path): 
    mapping_dic = {
        "ATB_TORCH_BUILT_IN_OP_OUTPUT_MAPPING": ATB_TORCH_BUILT_IN_OP_OUTPUT_MAPPING,
        "ATB_TORCH_CUSTOM_OP_OUTPUT_MAPPING": ATB_TORCH_CUSTOM_OP_OUTPUT_MAPPING,
    }
    mapping_file = os.path.join(mapping_file_path, "op_mapping_file.json")
    
    if os.path.exists(mapping_file):
        mapping_file = load_file_to_read_common_check(mapping_file)
        with ms_open(mapping_file, "r", max_size=TENSOR_MAX_SIZE) as file:
            file_content = json.load(file)

        if validate_json(file_content):
            for k, v in file_content.items():
                mapping_dic["ATB_TORCH_CUSTOM_OP_OUTPUT_MAPPING"][k] = v
            msg = f"Using user-specified op_mapping from file: {mapping_file}"
            logger.info(msg)
        else:
            msg = f"Invalid op_mapping file: {mapping_file}"
            logger.error(msg)
    else:
        logger.debug("Using built-in op_mapping")
    return mapping_dic


def cmp_torch_atb(torch_model_topo_file, cmp_paths, mapping_file_path, cmp_level="layer"):
    golden_path, my_path, output_path = cmp_paths
    try:
        path_index = -2 if cmp_level == "layer" else -1
        pid = str(my_path.split("/")[path_index].split("_")[1])
    except IndexError:
        pid = ""
        msg = f"Cannot parse the right pid from my_path! my_path: {my_path}"
        logger.error(msg)
    
    csv_file_path = ""
    atb_model_topo_file_path = os.path.join(my_path, "../../.." if cmp_level == "layer" else "../..", "model", pid)
    if os.path.exists(atb_model_topo_file_path):
        atb_model_topo_name = os.listdir(atb_model_topo_file_path)[0]
        atb_model_topo_file = os.path.join(atb_model_topo_file_path, atb_model_topo_name)
        if os.path.exists(atb_model_topo_file) and cmp_level == "layer":
            mapping_dic = load_mapping(mapping_file_path)
            data_info = {
                "golden_json": torch_model_topo_file,
                "my_json": atb_model_topo_file,
                "torch_tensor_path": golden_path,
                "atb_tensor_path": my_path,
            }
            csv_file_path = cmp_torch_atb_model(data_info, output_path, mapping_dic)
        elif cmp_level == "token":
            compared_results = []

            for token_id in os.listdir(my_path):
                torch_token_path = os.path.join(golden_path, str(token_id))
                atb_token_path = os.path.join(my_path, str(token_id))

                if not os.path.exists(torch_token_path) or not os.path.exists(atb_token_path):
                    continue

                if not os.path.isdir(torch_token_path) or not os.path.isdir(atb_token_path):
                    continue

                compare_result = cmp_torch_atb_token(torch_token_path, atb_token_path, token_id)

                compared_results.extend(compare_result)
                
            csv_file_path = save_compare_result_to_csv(compared_results, output_path)
        else:
            msg = f"Cannot find atb model file: {atb_model_topo_file}"
            logger.error(msg)
    else:
        msg = f"Cannot find atb model file path: {atb_model_topo_file_path}"
        logger.error(msg)
    return csv_file_path
