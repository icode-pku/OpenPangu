# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

from .utils import logger
from .constants import US_PER_MS

CSV_COLUMNS = ["name", "during_time", "pid", "tid", "start_time", "end_time", "rid", "start_datetime", 
               "end_datetime", "batch_type", "batch_size", "rid_list", "token_id_list"]

RENAMED_COLUMNS = {
        "start_time": "start_time(ms)",
        "end_time": "end_time(ms)",
        "during_time": "during_time(ms)"
    }

PREFILL_NAME = "Prefill"
DECODE_NAME = "Decode"


def get_statistics_data(framework_df, filter_name, name):
    if framework_df.empty:
        logger.warning(f"{name}: The dataframe is empty, no csv file create")
        return framework_df
    
    time_columns = {
        "during_time(ms)": lambda x: x / US_PER_MS,
        "start_time(ms)": lambda x: x // US_PER_MS,
        "end_time(ms)": lambda x: x // US_PER_MS
    }

    # 批量转换时间单位
    for col, func in time_columns.items():
        if col in framework_df.columns:
            framework_df[col] = func(framework_df[col])
        else:
            logger.warning(f"Column '{col}' not found for time conversion.")

    mask = framework_df["name"] == filter_name
    if not mask.any():
        logger.warning(f"'{filter_name}' not found in the dataframe.")
        return framework_df
    filter_indices = framework_df[mask].index
    if len(filter_indices) == 1:
        start_index = filter_indices[-1]
        end_index = None
    else:
        max_interval_index = max(
            range(len(filter_indices) - 1),
            key=lambda i: filter_indices[i + 1] - filter_indices[i]
        )
        start_index = filter_indices[max_interval_index]
        end_index = filter_indices[max_interval_index + 1] - 1
    stats = ["max", "min", "mean", "std"]
    grouped = framework_df.groupby("name")["during_time(ms)"]
    for stat in stats:
        framework_df[stat] = grouped.transform(stat)
    # 处理标准差为 NaN 的情况
    framework_df["std"] = framework_df["std"].fillna(0)
    # 重新排列列顺序
    for idx, stat in enumerate(stats, start=2):
        framework_df.insert(idx, stat, framework_df.pop(stat))
    
    col_limit = 10 if name == DECODE_NAME else 11
    framework_df = framework_df.iloc[:, :col_limit]

    return framework_df[start_index: end_index]


def get_service_type(framework_df):
    from ..split_processor import VllmProcessor, MindIEProcessor, MindIEProcessorV2
    result_service = MindIEProcessor()
    name_set = set(list(framework_df["name"]))
    if "deserializeExecuteResponse" not in name_set:
        if "SerializeRequests" in name_set:
            result_service = MindIEProcessorV2()
        else:
            result_service = VllmProcessor()
    return result_service