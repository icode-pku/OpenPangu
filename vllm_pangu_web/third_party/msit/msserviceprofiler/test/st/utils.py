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

import subprocess
import os
import logging
import pytest
import pandas as pd

COMMAND_SUCCESS = 0


def execute_cmd(cmd):
    logging.info('Execute command:%s' % " ".join(cmd))
    completed_process = subprocess.run(cmd, shell=False, stderr=subprocess.PIPE)
    if completed_process.returncode != COMMAND_SUCCESS:
        logging.error(completed_process.stderr.decode())
    return completed_process.returncode


def check_column_actual(actual_columns, expected_columns, context):
    """检查实际列名是否与预期列名一致"""
    for col in expected_columns:
        if col not in actual_columns:
            logging.error(f"在 {context} 中未找到预期列名: {col}")
            return False
    return True


def check_row(df, expected_columns, numeric_columns):
    """检查数据框中Metric列数据类型和指定列数据是否为数字"""
    # 检查Metric列的数据类型是否为字符串
    for row_index in df.index:
        try:
            value = df.at[row_index, 'Metric']
            if not isinstance(value, str):
                logging.error(f"在Metric列的第{row_index}行，值 '{value}' 不是字符串类型")
                return False
        except KeyError:
            logging.error(f"数据框中不存在 'Metric' 列")
            return False

    # 检查其他列的数据是否为数字
    for column in numeric_columns:
        if column not in df.columns:
            logging.error(f"数据框中不存在 {column} 列")
            continue
        for row_index in df.index:
            try:
                cell_value = df.at[row_index, column]
                float(cell_value)
            except (ValueError, KeyError):
                logging.error(
                    f"在 {column} 列的第 {row_index} 行，值 {cell_value} 不是有效的数字")
                return False
    return True


def check_no_empty_lines_before_first_line(dataframe, context=""):
    empty_line = 0
    # 检查是否有空行
    for _, row in dataframe.iterrows():
        if row.isnull().all():
            empty_line += 1
        else:
            break
    
    pytest.assume(empty_line == 0, f"{context} table has {empty_line} empty lines before first line.")


def check_no_empty_lines_between_first_last_line(dataframe, context=""):
    # 计算非空行的数量
    empty_rows = dataframe.eq('').all(axis=1)
    num_empty_rows = empty_rows.sum()
    pytest.assume(num_empty_rows == 0, f"{context} table has empty lines.")


def check_during_time(dataframe, context=""):
    # 检查所需列是否存在于数据框中
    required_columns = ['end_time(ms)', 'start_time(ms)', 'during_time(ms)']
    for col in required_columns:
        if col not in dataframe.columns:
            logging.error(f"The column {col} not found in {context}.")
            return False

    # 检查during_time是否正确
    for index, row in dataframe.iloc[:-1].iterrows():
        end_time = row['end_time(ms)']
        start_time = row['start_time(ms)']
        during_time = row['during_time(ms)']
        # 计算 end_time - start_time 与 during_time 的差值
        diff = abs((end_time - start_time) - during_time)
        if diff > 1:
            logging.error(f"In row {index} of {context}, the during_time is not correct.")
            return False

    return True


def check_split_csv_content(output_path, csv_file_name):
    # 校验该路径下是否正确生成csv文件，以及文件内容
    csv_file = os.path.join(output_path, csv_file_name)
    pytest.assume(os.path.exists(csv_file), f"CSV file not found: {csv_file}")
    task_name = os.path.splitext(csv_file_name)[0]
    expected_header = ['name', 'during_time(ms)', 'max', 'min', 'mean', 'std', \
                       'pid', 'tid', 'start_time(ms)', 'end_time(ms)']
    if task_name == 'prefill':
        expected_header.append('rid')
    df = pd.read_csv(csv_file)
    # 检查列名是否正确
    result = check_column_actual(df.columns.tolist(), expected_header, context=csv_file_name)
    pytest.assume(result, f"{csv_file_name} column check failed")
    # 检查是否存在空行
    check_no_empty_lines_before_first_line(df, context=csv_file_name)
    check_no_empty_lines_between_first_last_line(df, context=csv_file_name)
    # 检查执行时间是否正确
    result = check_during_time(df, context=csv_file_name)
    pytest.assume(result, f"{csv_file_name} execution time validation failed")