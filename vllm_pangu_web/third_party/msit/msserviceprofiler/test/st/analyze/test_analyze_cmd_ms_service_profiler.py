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

import os
import json
import shutil
import sqlite3
from unittest import TestCase
import ast
import pytest
import pandas as pd
from jsonschema import validate, ValidationError
from st.utils import execute_cmd, check_column_actual, check_row
from msserviceprofiler.msguard.security import open_s


def check_csv_content(output_path, csv_file_name, expected_csv_columns, numeric_columns):
    csv_file = os.path.join(output_path, csv_file_name)
    # 检查文件是否存在
    assert os.path.exists(csv_file), f"文件 {csv_file} 不存在"
    assert os.path.isfile(csv_file), f"{csv_file} 不是一个有效的文件"
    df = pd.read_csv(csv_file)
    actual_columns = df.columns.tolist()

    check_column_actual(actual_columns, expected_csv_columns, context=csv_file_name)

    return check_row(df, expected_csv_columns, numeric_columns)


def check_kvcache_csv_content(output_path, csv_file_name):
    expected_csv_columns = [
        'domain', 'rid', 'timestamp(ms)',
        'name', 'device_kvcache_left'
    ]
    csv_file = os.path.join(output_path, csv_file_name)
    # 检查文件是否存在
    assert os.path.exists(csv_file)
    assert os.path.isfile(csv_file)

    df = pd.read_csv(csv_file)
    actual_columns = df.columns.tolist()
    check_column_actual(actual_columns, expected_csv_columns, context=csv_file_name)

    def is_whole_number(n):
        if n == int(n):
            return True
        else:
            return False

    # 定义一个函数，用于检查res_list的格式
    def check_rows(df, row_index, columns):
        for column in columns:
            if not is_whole_number(df.iloc[row_index][column]):
                raise AssertionError(f"{row_index}行的{column}不是整数")

    # 检查数据框的第一行和最后一行的特定列
    rows_to_check = [0, -1]
    columns_to_check = ['device_kvcache_left']
    for row_index in rows_to_check:
        if df.iloc[row_index]['name'] != 'allocate':
            for column in columns_to_check:
                check_rows(df, row_index, [column])


def check_batch_csv_content(output_path, csv_file_name):
    # 校验该路径下是否正确生成batch_data的csv文件，以及文件内容
    csv_file = os.path.join(output_path, csv_file_name)
    assert os.path.exists(csv_file)
    assert os.path.isfile(csv_file)
    expected_header = ['name', 'res_list', 'start_time(ms)', 'end_time(ms)', 'batch_size', \
                       'batch_type', 'during_time(ms)']
    df = pd.read_csv(csv_file)
    # 检查列名是否正确
    check_column_actual(df.columns.tolist(), expected_header, context='batch.csv')

    # 定义一个函数，用于检查res_list的格式
    def is_valid_res_list(res_list_str):
        # 将字符串转换为列表
        res_list = ast.literal_eval(res_list_str)
        # 检查res_list是否是一个列表，每个元素都是字典，且字典包含'rid'和'iter'这两个键
        return all(isinstance(item, dict) and 'rid' in item and 'iter' in item for item in res_list)

    # 检查数据框的第一行和最后一行的特定列
    rows_to_check = [0, -1]
    columns_to_check = ['res_list']
    for row_index in rows_to_check:
        for column in columns_to_check:
            assert is_valid_res_list(df.iloc[row_index][column]), f"{row_index}行的{column}格式不正确"


def check_request_csv_content(output_path, csv_file_name):
    # 校验该路径下是否正确生成req_data的csv文件，以及文件内容
    csv_file = os.path.join(output_path, csv_file_name)
    assert os.path.exists(csv_file)
    assert os.path.isfile(csv_file)
    df = pd.read_csv(csv_file)
    expected_header = ['http_rid', 'start_time(ms)', 'recv_token_size', 'reply_token_size', \
                       'execution_time(ms)', 'queue_wait_time(ms)']
    check_column_actual(df.columns.tolist(), expected_header, context='request.csv')

    def is_whole_number(n):
        if n == int(n):
            return True
        else:
            return False

    # 定义一个函数，用于检查数据框的某一行的特定列是否满足条件
    def check_rows(df, row_index, columns):
        for column in columns:
            if not is_whole_number(df.iloc[row_index][column]):
                raise AssertionError(f"{row_index}行的{column}不是整数")

    # 检查execution_time(ms)列有数据行
    rows_to_check = df[df['execution_time(ms)'].notna()]
    columns_to_check = ['recv_token_size', 'reply_token_size']
    for row_index, _ in rows_to_check.iterrows():
        for column in columns_to_check:
            check_rows(df, row_index, [column])


def check_pullkvcache_csv_content(csv_file):
    expected_csv_columns = [
        'domain', 'rank', 'rid', 'block_tables', 'batch_seq_len', 'during_time(ms)', \
        'start_datetime(ms)', 'end_datetime(ms)', 'start_time(ms)', 'end_time(ms)',
    ]
    # 检查文件是否存在
    assert os.path.exists(csv_file)
    assert os.path.isfile(csv_file)

    df = pd.read_csv(csv_file)
    actual_columns = df.columns.tolist()
    check_column_actual(actual_columns, expected_csv_columns, context=csv_file)


def check_has_vaild_table(cursor, table_name, columns_to_check):
    # 校验存在数据表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    table_exists = cursor.fetchone()
    assert table_exists is not None

    # 校验生成的列
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_in_table = [row[1] for row in cursor]
    pytest.assume(all(column in columns_in_table for column in columns_to_check))

    # 校验至少存在一行所有的列都不为空
    cursor.execute(f"SELECT * FROM {table_name}")
    data = cursor.fetchall()
    for row in data:
        if all(row):
            return
    pytest.assume(False)


def check_latency_db_content(output_path, db_file_name):
    # 校验db文件正常生成
    db_path = os.path.join(output_path, db_file_name)
    assert os.path.exists(db_path)

    # 校验时延数据表
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    columns_to_check = ['avg', 'p50', 'p90', 'p99', 'timestamp']
    check_has_vaild_table(cursor, 'decode_gen_speed', columns_to_check)
    check_has_vaild_table(cursor, 'first_token_latency', columns_to_check)
    check_has_vaild_table(cursor, 'prefill_gen_speed', columns_to_check)
    check_has_vaild_table(cursor, 'req_latency', columns_to_check)

    # 关闭连接
    conn.close()


def check_kvcache_db_content(output_path, db_file_name):
    db_file = os.path.join(output_path, db_file_name)
    expected_db_columns = [
        'rid',
        'name',
        'real_start_time(ms)',
        'device_kvcache_left',
        'kvcache_usage_rate'
    ]
    assert os.path.exists(db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('PRAGMA table_info("kvcache")')
    columns = cursor.fetchall()
    actual_columns = [column[1] for column in columns]

    check_column_actual(actual_columns, expected_db_columns, context=db_file_name)

    conn.close()


def check_req_status_db_content(output_path, db_file_name):
    from enum import Enum

    class ReqStatus(Enum):
        WAITING = 0
        PENDING = 1
        RUNNING = 2
        SWAPPED = 3
        RECOMPUTE = 4
        SUSPENDED = 5
        END = 6
        STOP = 7
        PREFILL_HOLD = 8

    # 校验db文件正常生成
    db_path = os.path.join(output_path, db_file_name)
    assert os.path.exists(db_path)

    # 获取数据表
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM request_status", conn)
    conn.close()

    # 校验列存在
    for col in ['timestamp', 'WAITING', 'PENDING', 'RUNNING']:
        assert col in df.columns.tolist()


def check_chrome_tracing_valid(output_path, file_name):
    trace_view_json = os.path.join(output_path, file_name)
    assert os.path.exists(trace_view_json), f"文件 {trace_view_json} 不存在"
    assert os.path.isfile(trace_view_json), f"{trace_view_json} 不是一个有效的文件"

    schema = {
        "type": "object",
        "properties": {
            "traceEvents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "ph": {"type": "string", "enum": ["X", "I", "C", "M", "s", "f", "t"]},
                        "ts": {"type": ["number", "string"],
                               "pattern": "^\\d+(\\.\\d+)?$"
                               },  # 时间戳，单位为微秒
                        "dur": {"type": "number", "minimum": 0},  # 持续时间，适用于 X 类型事件
                        "pid": {"type": "integer"},  # 进程 ID
                        "tid": {"type": ["string", "integer"]},
                        "id": {"type": "string"},  # 时间线事件的 ID
                        "cat": {"type": "string"},  # 分类
                        "args": {
                            "type": "object",
                            "additionalProperties": True  # args 可以是任意键值对
                        }
                    },
                    "required": ["name", "ph", "pid"],  # 必需字段
                    "additionalProperties": False  # 防止额外字段
                }
            }
        },
        "required": ["traceEvents"],  # 必需字段
        "additionalProperties": False  # 防止额外字段
    }
    with open_s(trace_view_json) as f:
        data = json.load(f)

    validate(instance=data, schema=schema)


def check_chrome_tracing_valid_ms_op(output_path, file_name):
    trace_view_json = os.path.join(output_path, file_name)
    assert os.path.exists(trace_view_json), f"文件 {trace_view_json} 不存在"
    assert os.path.isfile(trace_view_json), f"{trace_view_json} 不是一个有效的文件"

    schema = {
        "type": "object",
        "properties": {
            "traceEvents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "ph": {"type": "string", "enum": ["X", "I", "C", "M", "s", "f", "t"]},
                        "ts": {"type": ["number", "string"],
                               "pattern": "^\\d+(\\.\\d+)?$"
                               },  # 时间戳，单位为微秒
                        "dur": {"type": "number", "minimum": 0},  # 持续时间，适用于 X 类型事件
                        "pid": {"type": "integer"},  # 进程 ID
                        "tid": {"type": ["string", "integer"]},
                        "id": {"type": "string"},  # 时间线事件的 ID
                        "cat": {"type": "string"},  # 分类
                        "bp": {"type": "string"},
                        "args": {
                            "type": "object",
                            "additionalProperties": True  # args 可以是任意键值对
                        }
                    },
                    "required": ["name", "ph", "pid"],  # 必需字段
                    "additionalProperties": False  # 防止额外字段
                }
            }
        },
        "required": ["traceEvents"],  # 必需字段
        "additionalProperties": False  # 防止额外字段
    }
    with open_s(trace_view_json) as f:
        data = json.load(f)

    validate(instance=data, schema=schema)


def check_chrome_tracing_content_valid(output_path, file_name):
    trace_view_json = os.path.join(output_path, file_name)

    with open_s(trace_view_json, 'r', encoding='utf-8') as f:
        text = f.read()
    exist = ["Execute", "BatchSchedule"]
    for key in exist:
        pytest.assume(key in text, f"The chrome trace file shoule include {key}.")


class TestAnalyzeCmd(TestCase):
    ST_DATA_PATH = os.getenv("MS_SERVICE_PROFILER",
                             "/data/ms_service_profiler")
    INPUT_PATH_MSSERVICEPROFILER = "/tmp/server-smoke/latest_PD_competition_ms"
    INPUT_PATH_MSSERVICEPROFILER_OPERATOR = "/tmp/server-smoke/latest_PD_competition_ms_op"
    INPUT_PATH_PD_SEPARATE = os.path.join(ST_DATA_PATH, "input/analyze/latest_PD_split")
    OUTPUT_PATH = os.path.join(ST_DATA_PATH, "output/analyze")
    REQUEST_SUM_CSV = "request_summary.csv"
    BATCH_SUM_CSV = "batch_summary.csv"
    SERVICE_SUM_CSV = "service_summary.csv"
    KVCACHE_CSV = "kvcache.csv"
    BATCH_CSV = "batch.csv"
    REQUEST_CSV = "request.csv"
    PROFILER_DB = "profiler.db"
    CHROME_TRACE = "chrome_tracing.json"
    COMMAND_SUCCESS = 0
    ANALYZE_PROFILER = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
                                    "msserviceprofiler/__main__.py")

    def setUp(self):
        os.makedirs(self.OUTPUT_PATH, mode=0o750, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.OUTPUT_PATH)

    def test_analyze_ms_service_profiler_data(self):
        # 校验msserviceprofiler打点采集数据解析功能是否正常解析，校验输出文件及内容
        cmd = [
            "python", self.ANALYZE_PROFILER, "analyze",
            "--input-path", self.INPUT_PATH_MSSERVICEPROFILER,
            "--output-path", self.OUTPUT_PATH
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(
                True, msg="enable ms service profiler analyze task failed.")

        request_columns = ['Metric', 'Average',
                           'Max', 'Min', 'P50', 'P90', 'P99']
        request_numeric_columns = ['Average',
                                   'Max', 'Min', 'P50', 'P90', 'P99']

        service_columns = ['Metric', 'Value']
        service_numeric_columns = ['Value']

        with self.subTest("Check request_summary.csv content"):
            try:
                result = check_csv_content(
                    self.OUTPUT_PATH, self.REQUEST_SUM_CSV, request_columns, request_numeric_columns)
                self.assertTrue(result, f"检查 {self.REQUEST_SUM_CSV} 失败")
            except Exception as e:
                self.fail(f"检查 {self.REQUEST_SUM_CSV} 时发生异常: {e}")

        with self.subTest("Check batch_summary.csv content"):
            try:
                result = check_csv_content(
                    self.OUTPUT_PATH, self.BATCH_SUM_CSV, request_columns, request_numeric_columns)
                self.assertTrue(result, f"检查 {self.BATCH_SUM_CSV} 失败")
            except Exception as e:
                self.fail(f"检查 {self.BATCH_SUM_CSV} 时发生异常: {e}")

        with self.subTest("Check service_summary.csv content"):
            try:
                result = check_csv_content(
                    self.OUTPUT_PATH, self.SERVICE_SUM_CSV, service_columns, service_numeric_columns)
                self.assertTrue(result, f"检查 {self.SERVICE_SUM_CSV} 失败")
            except Exception as e:
                self.fail(f"检查 {self.SERVICE_SUM_CSV} 时发生异常: {e}")

        with self.subTest("Check chrome_tracing.json content"):
            try:
                check_chrome_tracing_valid(self.OUTPUT_PATH, self.CHROME_TRACE)
                check_chrome_tracing_content_valid(self.OUTPUT_PATH, self.CHROME_TRACE)
            except Exception as e:
                self.fail(f"检查 {self.CHROME_TRACE} 时发生异常: {e}")

        with self.subTest("Check profiler.db content"):
            try:
                check_latency_db_content(self.OUTPUT_PATH, self.PROFILER_DB)
                check_kvcache_db_content(self.OUTPUT_PATH, self.PROFILER_DB)
                check_req_status_db_content(self.OUTPUT_PATH, self.PROFILER_DB)
            except Exception as e:
                self.fail(f"检查 {self.PROFILER_DB} 时发生异常: {e}")

        with self.subTest("Check kvcache.csv content"):
            try:
                check_kvcache_csv_content(self.OUTPUT_PATH, self.KVCACHE_CSV)
            except Exception as e:
                self.fail(f"检查 {self.KVCACHE_CSV} 时发生异常: {e}")

        with self.subTest("Check batch.csv content"):
            try:
                check_batch_csv_content(self.OUTPUT_PATH, self.BATCH_CSV)
            except Exception as e:
                self.fail(f"检查 {self.BATCH_CSV} 时发生异常: {e}")

        with self.subTest("Check request.csv content"):
            try:
                check_request_csv_content(self.OUTPUT_PATH, self.REQUEST_CSV)
            except Exception as e:
                self.fail(f"检查 {self.REQUEST_CSV} 时发生异常: {e}")

    def test_parse_data_in_pd_separate(self):
        # 校验msserviceprofiler打点PD分离数据解析功能是否正常解析，校验输出文件及内容
        cmd = ["python", self.ANALYZE_PROFILER, "analyze", "--input-path", self.INPUT_PATH_PD_SEPARATE, \
               "--output-path", self.OUTPUT_PATH]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(True, msg="enable ms service profiler analyze task failed.")

        with self.subTest("Check pullkvcache csv content"):
            check_pullkvcache_csv_content(os.path.join(self.OUTPUT_PATH, "pd_split_kvcache.csv"))

    def test_analyze_ms_service_operator_profiler_data(self):
        # 校验msserviceprofiler_operator打点采集数据解析功能是否正常解析，校验输出文件及内容
        cmd = [
            "python", self.ANALYZE_PROFILER, "analyze",
            "--input-path", self.INPUT_PATH_MSSERVICEPROFILER_OPERATOR,
            "--output-path", self.OUTPUT_PATH
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(
                True, msg="enable ms service profiler analyze task failed.")

        request_columns = ['Metric', 'Average',
                           'Max', 'Min', 'P50', 'P90', 'P99']
        request_numeric_columns = ['Average',
                                   'Max', 'Min', 'P50', 'P90', 'P99']

        service_columns = ['Metric', 'Value']
        service_numeric_columns = ['Value']

        with self.subTest("Check request_summary.csv content"):
            try:
                result = check_csv_content(
                    self.OUTPUT_PATH, self.REQUEST_SUM_CSV, request_columns, request_numeric_columns)
                self.assertTrue(result, f"检查 {self.REQUEST_SUM_CSV} 失败")
            except Exception as e:
                self.fail(f"检查 {self.REQUEST_SUM_CSV} 时发生异常: {e}")

        with self.subTest("Check batch_summary.csv content"):
            try:
                result = check_csv_content(
                    self.OUTPUT_PATH, self.BATCH_SUM_CSV, request_columns, request_numeric_columns)
                self.assertTrue(result, f"检查 {self.BATCH_SUM_CSV} 失败")
            except Exception as e:
                self.fail(f"检查 {self.BATCH_SUM_CSV} 时发生异常: {e}")

        with self.subTest("Check service_summary.csv content"):
            try:
                result = check_csv_content(
                    self.OUTPUT_PATH, self.SERVICE_SUM_CSV, service_columns, service_numeric_columns)
                self.assertTrue(result, f"检查 {self.SERVICE_SUM_CSV} 失败")
            except Exception as e:
                self.fail(f"检查 {self.SERVICE_SUM_CSV} 时发生异常: {e}")

        with self.subTest("Check chrome_tracing.json content"):
            try:
                check_chrome_tracing_valid_ms_op(self.OUTPUT_PATH, self.CHROME_TRACE)
                check_chrome_tracing_content_valid(self.OUTPUT_PATH, self.CHROME_TRACE)
            except Exception as e:
                self.fail(f"检查 {self.CHROME_TRACE} 时发生异常: {e}")

        with self.subTest("Check profiler.db content"):
            try:
                check_latency_db_content(self.OUTPUT_PATH, self.PROFILER_DB)
                check_kvcache_db_content(self.OUTPUT_PATH, self.PROFILER_DB)
                check_req_status_db_content(self.OUTPUT_PATH, self.PROFILER_DB)
            except Exception as e:
                self.fail(f"检查 {self.PROFILER_DB} 时发生异常: {e}")

        with self.subTest("Check kvcache.csv content"):
            try:
                check_kvcache_csv_content(self.OUTPUT_PATH, self.KVCACHE_CSV)
            except Exception as e:
                self.fail(f"检查 {self.KVCACHE_CSV} 时发生异常: {e}")

        with self.subTest("Check batch.csv content"):
            try:
                check_batch_csv_content(self.OUTPUT_PATH, self.BATCH_CSV)
            except Exception as e:
                self.fail(f"检查 {self.BATCH_CSV} 时发生异常: {e}")

        with self.subTest("Check request.csv content"):
            try:
                check_request_csv_content(self.OUTPUT_PATH, self.REQUEST_CSV)
            except Exception as e:
                self.fail(f"检查 {self.REQUEST_CSV} 时发生异常: {e}")