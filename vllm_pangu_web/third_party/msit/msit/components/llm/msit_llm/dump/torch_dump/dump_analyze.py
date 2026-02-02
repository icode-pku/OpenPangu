# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
import csv
from pathlib import Path
import torch
from components.utils.file_open_check import ms_open
from msit_llm.common.constant import get_ait_dump_path


def dump_analyze(feat, feat_path: str, module_name, dump_type, dump_path) -> None:
    #提取dump_type_name作为字符串
    dump_type_name = str(dump_type).split('of ')[-1].split('(')[0]

    #获取存放csv文件的路径
    cache_path = os.path.join(dump_path, get_ait_dump_path())

    #定义csv文件名
    csv_filename = "tensor_analyze.csv"
    output_csv_path = os.path.join(cache_path, csv_filename)

    #打开csv文件，用追加数据的方式进行写入
    with ms_open(output_csv_path, mode="a") as file:
        writer = csv.writer(file)

        file_is_empty = os.path.getsize(output_csv_path) == 0
        if file_is_empty:
            writer.writerow([
                'Node Name', 'Node Type', 'Tensor Path', 'dtype', 'shape', 'max', 'min', 'variance', 'mean'
            ])

        data_tensor = feat.float()

        writer.writerow([
            module_name,     
            dump_type_name,
            feat_path,
            feat.dtype,
            data_tensor.shape,
            torch.max(data_tensor).item(),
            torch.min(data_tensor).item(),
            torch.var(data_tensor).item(),
            torch.mean(data_tensor).item(),
        ]) 