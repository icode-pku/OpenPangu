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

import torch
from tqdm import tqdm

from safetensors.torch import load_file
from components.utils.constants import FileCheckConst
from components.utils.file_utils import check_and_get_real_path
from components.utils.util import safe_torch_load
from msit_llm.common.log import logger
from msit_llm.common.utils import load_file_to_read_common_check
from msit_llm.common.constant import CSV_CMP_WEIGHT_HEADER
from msit_llm.compare.cmp_utils import (
    save_compare_result_to_csv,
    compare_data,
    set_tensor_basic_info_in_row_data,
)


def find_safetensors_files(golden_path):
    model_dir_path = os.path.abspath(golden_path)
    # 搜索给定目录下的所有文件，查找并存储safetensors文件路径
    # 如果没有找到safetensors文件就报错
    safetensors_file_list, bin_file_list = [], []
    for file in os.listdir(model_dir_path):
        safetensors_file_path = os.path.join(model_dir_path, file)
        if not os.path.isfile(safetensors_file_path):
            continue
        if file.endswith(".safetensors") and check_and_get_real_path(safetensors_file_path, FileCheckConst.READ_ABLE):
            safetensors_file_list.append(safetensors_file_path)
        if file.endswith(".bin") and check_and_get_real_path(safetensors_file_path, FileCheckConst.READ_ABLE):
            bin_file_list.append(safetensors_file_path)
    return safetensors_file_list if safetensors_file_list else bin_file_list


def dequant(weight, weight_offset, weight_scale):
    return (weight - weight_offset) * weight_scale


def compare_weight(gp_path, mp_path, output_path):
    gp_path_list = find_safetensors_files(gp_path)
    mp_path_list = find_safetensors_files(mp_path)
    if not gp_path_list or not mp_path_list:
        logger.error("No .safetensors files found in the directory.")
        raise FileNotFoundError("Invalid path")

    gathered_row_data = []
    sorted_gp_path_list = sorted(gp_path_list)

    mp_path_list[0] = load_file_to_read_common_check(mp_path_list[0])
    mp_dict = load_file(mp_path_list[0])

    for g_path in tqdm(sorted_gp_path_list, desc="Comparing"):
        g_path = load_file_to_read_common_check(g_path)
        if g_path.endswith(".safetensors"):
            gp_dict = load_file(g_path)
        elif g_path.endswith(".bin"):
            gp_dict = safe_torch_load(g_path, map_location="cpu")

        for ft_weight_key, ft_weight_value in gp_dict.items():

            if not ft_weight_key.endswith("weight"):
                continue
            int_weight_value = mp_dict.get(ft_weight_key, None)

            if int_weight_value is None or int_weight_value.dtype != torch.int8:
                continue

            weight_offset_key = ft_weight_key.replace("weight", "weight_offset")
            weight_scale_key = ft_weight_key.replace("weight", "weight_scale")
            weight_offset_value = mp_dict.get(weight_offset_key, None)
            weight_scale_value = mp_dict.get(weight_scale_key, None)

            if weight_offset_value is None or weight_scale_value is None:
                continue

            dequant_weight_value = dequant(int_weight_value, weight_offset_value, weight_scale_value)
            row_data_basic = set_tensor_basic_info_in_row_data(ft_weight_value, dequant_weight_value)
            row_data = compare_data(ft_weight_value, dequant_weight_value)
            row_data.update(row_data_basic)
            row_data.update({"weight_name": ft_weight_key})
            gathered_row_data.append(row_data)

    return save_compare_result_to_csv(gathered_row_data, output_path, columns=CSV_CMP_WEIGHT_HEADER)