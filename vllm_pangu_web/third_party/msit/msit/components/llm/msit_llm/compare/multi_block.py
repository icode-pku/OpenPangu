# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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

import torch
from msit_llm.common.log import logger
from msit_llm.compare.cmp_utils import BasicDataInfo, fill_row_data, read_data, read_data_statistics


def multi_block_cmp(atb_nodes, torch_nodes, my_root_node, atb_tensor_path, torch_tensor_path):
    compared_result = []
    for atb_node, torch_node in zip(atb_nodes, torch_nodes):
        if atb_node.op_type == "LinearOperation" and not atb_node.op_param.get("hasBias"):
            next_sibling_node = my_root_node.get_next_sibling_node(atb_node)
            # 当有些算子如ParallelLinearBaseV2，是将w*x+b的操作拆分成两个算子，linear+add，而torch中使用一个算子Linear实现，
            # 因此add node的输出映射的是torch中Linear的输出
            if next_sibling_node and next_sibling_node.op_type == "ElewiseOperation" \
                    and next_sibling_node.op_param.get('elewiseType') == 8:
                atb_node = next_sibling_node
        # 1. get tensor_datas
        atb_node_tensor_path = atb_node.tensor_path
        torch_node_tensor_path = torch_node.tensor_path
        atb_node_tensor_path, atb_tensor_datas = get_multi_tensor_paths(atb_tensor_path, atb_node_tensor_path,
            tensor_sub_dir=os.path.join("after", "outtensor0.bin"))
        torch_node_tensor_path, torch_tensor_datas = get_multi_tensor_paths(torch_tensor_path, torch_node_tensor_path,
            tensor_sub_dir="output.pth")
        # 存在torch数据或者atb数据不存在，跳过当前比对
        if not atb_tensor_datas or not torch_tensor_datas:
            continue
        # 2. concat tensor_datas
        dim_atb = get_cat_dim(atb_tensor_datas, torch_tensor_datas)
        dim_torch = get_cat_dim(torch_tensor_datas, atb_tensor_datas)
        atb_tensor_data = atb_tensor_datas[0] if dim_atb == -1 else torch.cat(atb_tensor_datas, dim_atb)
        torch_tensor_data = torch_tensor_datas[0] if dim_torch == -1 else torch.cat(torch_tensor_datas, dim_torch)
        # 3. compare tensor_datas
        data_info = BasicDataInfo(torch_node_tensor_path, atb_node_tensor_path)
        row_data = fill_row_data(data_info, atb_tensor_data, torch_tensor_data)
        compared_result.append(row_data)

    return compared_result


def get_multi_tensor_paths(data_path, node_path, tensor_sub_dir):
    tensor_device_name = os.path.basename(os.path.abspath(os.path.join(data_path, "..")))  # 0_npu_pid, 1_npu_pid
    device_tensor_paths = sorted(os.listdir(os.path.join(data_path, "..", "..")))
    cur_tensor_path = os.path.abspath(os.path.join(node_path, tensor_sub_dir))
    if not os.path.exists(cur_tensor_path):
        msg = f"golden tensor path: {cur_tensor_path} is not exist."
        logger.debug(msg)
        return None, None
    device_tensor_paths = [cur_tensor_path.replace(tensor_device_name, ii) for ii in device_tensor_paths]
    tensor_datas = []
    for tensor_path in device_tensor_paths:
        if os.path.exists(tensor_path):
            tensor_datas.append(read_data(tensor_path))
    return cur_tensor_path, tensor_datas


def get_cat_dim(atb_tensor_datas, torch_tensor_datas):
    for dim, (atb_tensor_size, torch_tensor_size) in enumerate(
            zip(atb_tensor_datas[0].shape, torch_tensor_datas[0].shape)):
        multi_torch_size = torch_tensor_size * len(torch_tensor_datas)
        multi_atb_size = atb_tensor_size * len(atb_tensor_datas)
        if multi_torch_size == multi_atb_size and torch_tensor_size != atb_tensor_size:
            return dim
        if multi_atb_size == torch_tensor_size:
            return dim
    return -1


def get_multi_statistics_paths(data_path, node_path, tensor_sub_dir):
    tensor_device_name = os.path.basename(os.path.abspath(os.path.join(data_path, "..")))
    device_tensor_paths = sorted(os.listdir(os.path.join(data_path, "..", "..")))
    cur_tensor_path = os.path.abspath(os.path.join(node_path, tensor_sub_dir))
    if not os.path.exists(cur_tensor_path):
        msg = f"golden tensor path: {cur_tensor_path} is not exist."
        logger.debug(msg)
        return None, None
    device_tensor_paths = [cur_tensor_path.replace(tensor_device_name, ii) for ii in device_tensor_paths]
    for tensor_path in device_tensor_paths:
        if os.path.exists(tensor_path):
            tensor_datas = read_data_statistics(tensor_path)
    return cur_tensor_path, tensor_datas