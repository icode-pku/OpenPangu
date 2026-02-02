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

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from msserviceprofiler.ms_service_profiler_ext.exporters.exporter_summary import (
    is_contained_valid_iter_info,
    process_batch_record,
    calculate_statistics,
    convert_map_to_dataframe,
    process_req_record,
    process_rid_token_list,
    calculate_request_metrics,
    gen_exporter_results,
    process_each_record,
    save_dataframe_to_csv,
    calculate_batch_metrics,
    get_new_total_time,
    get_new_ttft_wait_time,
    ExporterSummary
)
from msserviceprofiler.ms_service_profiler_ext.common.csv_fields import (
    RequestCSVFields, BatchCSVFields, ServiceCSVFields
)


class TestExporterSummaryFunctions(unittest.TestCase):
    def setUp(self):
        self.sample_batch_map = {}
        self.sample_req_map = {"1001": {"token_id": {}}}
        self.valid_rid_list = [1, 2]
        self.valid_iter_list = [0, 1]
        self.req_map = {}

    def test_is_contained_valid_iter_info_normal(self):
        self.assertTrue(
            is_contained_valid_iter_info(self.valid_rid_list, self.valid_iter_list),
            "Valid iteration information should be verified.",
        )

    def test_is_contained_valid_iter_info_edge_cases(self):
        self.assertFalse(
            is_contained_valid_iter_info([], self.valid_iter_list), "An empty rid list should return False."
        )
        self.assertFalse(
            is_contained_valid_iter_info([1], self.valid_iter_list), "Inconsistent lengths should return False."
        )
        self.assertFalse(
            is_contained_valid_iter_info(None, self.valid_iter_list), "If rid_list is None, it should return False."
        )
        self.assertFalse(
            is_contained_valid_iter_info(self.valid_rid_list, None), "If token_id_list is None, it should return False."
        )

    def test_process_batch_record_multiple_types(self):
        prefill_record = {
            "batch_type": "Prefill",
            "rid_list": [1001],
            "batch_size": 8,
            "during_time": 1_500_000,  # 1.5秒
        }
        process_batch_record(self.sample_batch_map, prefill_record)
        prefill_key = f"prefill_{str(prefill_record['rid_list'])}"
        self.assertIn(prefill_key, self.sample_batch_map, f"The key {prefill_key} was not found in sample_batch_map.")
        self.assertEqual(self.sample_batch_map[prefill_key]["prefill_batch_num"], 8)
        self.assertAlmostEqual(self.sample_batch_map[prefill_key]["prefill_exec_time (ms)"], 1500.0)

        # 测试Decode类型
        decode_record = {"batch_type": "Decode", "rid_list": [1002], "batch_size": 4, "during_time": 500_000}  # 0.5秒
        process_batch_record(self.sample_batch_map, decode_record)
        decode_key = f"decode_{str(decode_record['rid_list'])}"
        self.assertIn(decode_key, self.sample_batch_map, f"The key {decode_key} was not found in sample_batch_map.")
        self.assertEqual(self.sample_batch_map[decode_key]["decode_batch_num"], 4)
        self.assertAlmostEqual(self.sample_batch_map[decode_key]["decode_exec_time (ms)"], 500.0)
        unknown_record = {"batch_type": "Unknown", "rid_list": [1003], "batch_size": 2, "during_time": 200_000}
        process_batch_record(self.sample_batch_map, unknown_record)
        self.assertEqual(len(self.sample_batch_map), 2, "Unknown batch_type should not add new items.")

    def test_calculate_statistics_comprehensive(self):
        """测试全面的统计计算"""
        test_cases = [
            ([10, 20, 30], {"avg": 20, "p50": 20, "p90": 28, "p99": 29.8}),
            ([5], {"avg": 5, "p50": 5, "p90": 5, "p99": 5}),
            ([], {"avg": np.nan, "p50": np.nan, "p90": np.nan, "p99": np.nan}),
        ]

        for data, expected in test_cases:
            with self.subTest(data=data):
                result = calculate_statistics(data)
                if data:
                    self.assertAlmostEqual(result["avg"], expected["avg"], places=1)
                    self.assertAlmostEqual(result["p50"], expected["p50"], places=1)
                    self.assertAlmostEqual(result["p90"], expected["p90"], places=1)
                else:
                    self.assertTrue(np.isnan(result["avg"]))

        non_numeric_data = [10, "a", 30]
        result = calculate_statistics(non_numeric_data)
        self.assertTrue(np.isnan(result["avg"]), "If it contains non - numeric elements, it should return NaN.")

    def test_convert_map_to_dataframe_detailed(self):
        map_data = {
            "latency": {"avg": 15.5, "max": 30, "min": 5, "p50": 15, "p90": 27, "p99": 29},
            "throughput": {"avg": 100, "max": 200, "min": 50, "p50": 100, "p90": 180, "p99": 198},
        }
        df = convert_map_to_dataframe(map_data, include_stats=1)

        self.assertListEqual(df.index.tolist(), [0, 1])
        self.assertListEqual(df.columns.tolist(), ["Metric", "Average", "Max", "Min", "P50", "P90", "P99"])

        self.assertAlmostEqual(df.loc[0, "Average"], 15.5)
        self.assertEqual(df.loc[1, "Max"], 200)

        map_data_speed = {"generate_token_speed (token/s)": 123.456789}
        df_speed = convert_map_to_dataframe(map_data_speed, include_stats=0)
        self.assertListEqual(df_speed.columns.tolist(), ["Metric", "Value"])
        self.assertEqual(df_speed.loc[0, "Metric"], "generate_token_speed (token/s)")
        self.assertEqual(df_speed.loc[0, "Value"], "123.45678900")

    def test_full_request_lifecycle_metrics(self):
        req_map = {
            "1001": {
                "httpReq_start": 1_630_000_000_000,
                "httpRes_end": 1_630_000_002_000,
                "first_token_latency": 50_000,
                "exec_time": 100_000,
                "input_token_num": 10,
                "generated_token_num": 20,
                "token_id": {0: 1_630_000_000_500, 1: 1_630_000_001_000},
                "req_waiting_time": 10_000,
                "req_pending_time": 20_000,
            }
        }

        req_view, total_map = calculate_request_metrics(req_map)

        self.assertEqual(len(req_view), 1)
        request_metrics = req_view[0]

        self.assertEqual(request_metrics["req_id"], "1001")
        self.assertAlmostEqual(request_metrics["exec_time"], 100.0)
        self.assertAlmostEqual(request_metrics["first_token_latency"], 50.0)
        self.assertEqual(request_metrics["subsequent_token_latency"], [0.5])
        self.assertEqual(request_metrics["input_token_num"], 10)
        self.assertEqual(request_metrics["generated_token_num"], 20)

        self.assertEqual(total_map["total_input_token_num"], 10)
        self.assertEqual(total_map["total_generated_token_num"], 20)

        total_exec_time = (1_630_000_002_000 - 1_630_000_000_000) / 1000000
        generate_token_speed = 20 / total_exec_time
        generate_all_token_speed = (10 + 20) / total_exec_time
        self.assertAlmostEqual(total_map["generate_token_speed (token/s)"], generate_token_speed, places=3)
        self.assertAlmostEqual(total_map["generate_all_token_speed (token/s)"], generate_all_token_speed, places=3)

    def test_token_throughput_calculation(self):

        http_req_start = 1735124805384149
        token_id_0 = 1735124805784100
        token_id_1 = 1735124806061010
        http_res_end = 1735124815365194
        req_map = {
            "1": {
                "httpReq_start": http_req_start,
                "token_id": {
                    "0": token_id_0,
                    "1": token_id_1,
                },
                "req_waiting_time": 0.0,
                "req_pending_time": 0.0,
                "req_exec_time": 1735124805384149.2,
                "input_token_num": 4.0,
                "first_token_latency": 416819.0,
                "exec_time": 8915139.0,
                "generated_token_num": 250.0,
                "httpRes_end": http_res_end,
            }
        }

        _, total_map = calculate_request_metrics(req_map)
        self.assertAlmostEqual(total_map["total_input_token_num"], 4.0)
        self.assertAlmostEqual(total_map["total_generated_token_num"], 250.0)
        total_exec_time = (http_res_end - http_req_start) / 1000000
        generate_token_speed = 250 / total_exec_time
        generate_all_token_speed = (4 + 250) / total_exec_time

        generate_token_speed = round(generate_token_speed, 4)
        generate_all_token_speed = round(generate_all_token_speed, 4)

        self.assertAlmostEqual(total_map["generate_token_speed (token/s)"], generate_token_speed, places=3)
        self.assertAlmostEqual(total_map["generate_all_token_speed (token/s)"], generate_all_token_speed, places=3)

    def test_process_each_record(self):
        record = {"name": "httpReq", "rid": "1002", "start_time": 1_630_000_000_000}
        process_each_record(self.sample_req_map, self.sample_batch_map, record)
        self.assertIn("1002", self.sample_req_map, "Request records should be processed.")

    @patch("pandas.DataFrame.to_csv")
    def test_save_dataframe_to_csv(self, mock_to_csv):
        map_data = {"latency": {"avg": 15.5, "max": 30, "min": 5, "p50": 15, "p90": 27, "p99": 29}}
        output = "/tmp/test_output"
        file_name = "test.csv"
        save_dataframe_to_csv(map_data, output, file_name)
        mock_to_csv.assert_called_once()

    def test_process_req_record(self):
        http_req_record = {"name": "httpReq", "rid": "0", "start_time": 1739276837290586.5}
        process_req_record(self.req_map, http_req_record)
        token_id_record = {
            "name": "modelExec",
            "rid": "0",
            "rid_list": ["0"],
            "token_id_list": ["0"],
            "end_time": 1739276837517605.2,
            "during_time": 142208.75,
        }
        process_req_record(self.req_map, token_id_record)
        http_res_record = {"name": "httpRes", "rid": "0", "end_time": 1739276841538271.0, "replyTokenSize=": 250.0}
        process_req_record(self.req_map, http_res_record)
        input_token_record = {"name": "modelExec", "rid": "0", "recvTokenSize=": 4.0, "during_time": 142208.75}
        process_req_record(self.req_map, input_token_record)
        self.assertEqual(len(self.req_map), 1)
        self.assertEqual(self.req_map["0"]["httpReq_start"], 1739276837290586.5)
        self.assertEqual(self.req_map["0"]["token_id"]["0"], 1739276837517605.2)
        self.assertTrue(self.req_map["0"]["is_complete"], True)
        self.assertEqual(self.req_map["0"]["input_token_num"], 4.0)
        self.assertEqual(self.req_map["0"]["generated_token_num"], 250.0)
        self.assertEqual(self.req_map["0"]["httpRes_end"], 1739276841538271.0)

        rid_for_list = [0, 1, 2]
        token_id_for_list = [0, 0, 0]
        process_rid_token_list(self.req_map, rid_for_list, token_id_for_list, input_token_record)

    def test_calculate_batch_metrics(self):
        batch_map = {
            "prefill_[1,2]": {BatchCSVFields.PREFILL_BATCH_NUM: 8, BatchCSVFields.PREFILL_EXEC_TIME: 1.5},
            "prefill_[3]": {BatchCSVFields.PREFILL_BATCH_NUM: 4, BatchCSVFields.PREFILL_EXEC_TIME: 0.8},
            "decode_[4,5]": {BatchCSVFields.DECODE_BATCH_NUM: 6, BatchCSVFields.DECODE_EXEC_TIME: 2.0},
            "decode_[6]": {BatchCSVFields.DECODE_BATCH_NUM: 2, BatchCSVFields.DECODE_EXEC_TIME: 0.5},
        }
        result = calculate_batch_metrics(batch_map)

        prefill_batch_stats = result[BatchCSVFields.PREFILL_BATCH_NUM]
        self.assertEqual(prefill_batch_stats["avg"], (8 + 4) / 2)
        self.assertEqual(prefill_batch_stats["p50"], 6.0)

        prefill_time_stats = result[BatchCSVFields.PREFILL_EXEC_TIME]
        self.assertAlmostEqual(prefill_time_stats["avg"], (1.5 + 0.8) / 2, places=2)

        decode_batch_stats = result[BatchCSVFields.DECODE_BATCH_NUM]
        self.assertEqual(decode_batch_stats["avg"], (6 + 2) / 2)

        decode_time_stats = result[BatchCSVFields.DECODE_EXEC_TIME]
        self.assertAlmostEqual(decode_time_stats["max"], 2.0)

        empty_result = calculate_batch_metrics({})
        for field in [
            BatchCSVFields.PREFILL_BATCH_NUM,
            BatchCSVFields.DECODE_BATCH_NUM,
            BatchCSVFields.PREFILL_EXEC_TIME,
            BatchCSVFields.DECODE_EXEC_TIME,
        ]:
            self.assertTrue(
                np.isnan(empty_result[field]["avg"]), f"When batch_map is empty, the avg of {field} should be NaN."
            )

        partial_batch_map = {
            "prefill_[7]": {BatchCSVFields.PREFILL_EXEC_TIME: 1.2},
            "decode_[8]": {
                BatchCSVFields.DECODE_BATCH_NUM: 3,
            },
        }
        partial_result = calculate_batch_metrics(partial_batch_map)

        self.assertEqual(partial_result[BatchCSVFields.PREFILL_BATCH_NUM]["avg"], 0.0)

        self.assertEqual(partial_result[BatchCSVFields.DECODE_EXEC_TIME]["max"], 0.0)

    def test_process_batch_record_update_existing_keys(self):
        prefill_record_1 = {"batch_type": "Prefill", "rid_list": [1001], "batch_size": 8, "during_time": 1_500_000}
        process_batch_record(self.sample_batch_map, prefill_record_1)
        prefill_key = f"prefill_{str(prefill_record_1['rid_list'])}"

        prefill_record_2 = {"batch_type": "Prefill", "rid_list": [1001], "batch_size": 4, "during_time": 800_000}
        process_batch_record(self.sample_batch_map, prefill_record_2)

        self.assertEqual(self.sample_batch_map[prefill_key][BatchCSVFields.PREFILL_BATCH_NUM], 4)  # 覆盖原值
        self.assertAlmostEqual(self.sample_batch_map[prefill_key][BatchCSVFields.PREFILL_EXEC_TIME], 1500 + 800)

        decode_record_1 = {"batch_type": "Decode", "rid_list": [2001], "batch_size": 6, "during_time": 900_000}  # 0.9秒
        process_batch_record(self.sample_batch_map, decode_record_1)
        decode_key = f"decode_{str(decode_record_1['rid_list'])}"

        decode_record_2 = {
            "batch_type": "Decode",
            "rid_list": [2001],  # 相同的 rid_list
            "batch_size": 3,
            "during_time": 300_000,  # 0.3秒
        }
        process_batch_record(self.sample_batch_map, decode_record_2)

        self.assertEqual(self.sample_batch_map[decode_key][BatchCSVFields.DECODE_BATCH_NUM], 3)  # 覆盖原值
        self.assertAlmostEqual(self.sample_batch_map[decode_key][BatchCSVFields.DECODE_EXEC_TIME], 900 + 300)

    def test_process_batch_record_edge_cases(self):
        # 空 rid_list 的 Prefill 记录
        prefill_record_empty_rid = {"batch_type": "Prefill", "rid_list": [], "batch_size": 8, "during_time": 1_500_000}
        process_batch_record(self.sample_batch_map, prefill_record_empty_rid)
        prefill_key = "prefill_[]"
        self.assertIn(prefill_key, self.sample_batch_map)
        self.assertEqual(self.sample_batch_map[prefill_key][BatchCSVFields.PREFILL_BATCH_NUM], 8)

        # 无效 batch_type
        invalid_record = {"batch_type": "InvalidType", "rid_list": [3001], "batch_size": 2, "during_time": 200_000}
        process_batch_record(self.sample_batch_map, invalid_record)
        self.assertEqual(len(self.sample_batch_map), 1, "无效 batch_type 不应添加新条目")

    def test_process_batch_record_mixed_calls(self):
        prefill_record = {"batch_type": "Prefill", "rid_list": [1001], "batch_size": 8, "during_time": 1_500_000}
        process_batch_record(self.sample_batch_map, prefill_record)
        prefill_key = f"prefill_{str(prefill_record['rid_list'])}"

        decode_record = {"batch_type": "Decode", "rid_list": [2001], "batch_size": 4, "during_time": 500_000}
        process_batch_record(self.sample_batch_map, decode_record)
        decode_key = f"decode_{str(decode_record['rid_list'])}"

        prefill_record_2 = {"batch_type": "Prefill", "rid_list": [1002], "batch_size": 6, "during_time": 1_200_000}
        process_batch_record(self.sample_batch_map, prefill_record_2)
        prefill_key_2 = f"prefill_{str(prefill_record_2['rid_list'])}"

        self.assertEqual(len(self.sample_batch_map), 3)
        self.assertIn(prefill_key, self.sample_batch_map)
        self.assertIn(decode_key, self.sample_batch_map)
        self.assertIn(prefill_key_2, self.sample_batch_map)


class TestGenExporterResults(unittest.TestCase):
    @patch('msserviceprofiler.ms_service_profiler_ext.exporters.exporter_summary.process_each_record')
    @patch('msserviceprofiler.ms_service_profiler_ext.exporters.exporter_summary.calculate_request_metrics')
    @patch('msserviceprofiler.ms_service_profiler_ext.exporters.exporter_summary.calculate_statistics')
    @patch('msserviceprofiler.ms_service_profiler_ext.exporters.exporter_summary.calculate_batch_metrics')
    def test_gen_exporter_results(self, mock_calculate_batch_metrics, mock_calculate_statistics,
                                   mock_calculate_request_metrics, mock_process_each_record):
                                  
        # 创建一个模拟的DataFrame
        all_data_df = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })

        # 设置模拟函数的返回值
        mock_calculate_request_metrics.return_value = ({}, {})
        mock_calculate_statistics.return_value = {}
        mock_calculate_batch_metrics.return_value = {}

        # 调用函数
        gen_exporter_results(all_data_df)

        # 验证函数是否被正确调用
        mock_process_each_record.assert_called()
        mock_calculate_request_metrics.assert_called()
        mock_calculate_statistics.assert_called()
        mock_calculate_batch_metrics.assert_called()


class TestGetNewTotalTime(unittest.TestCase):
    @patch('msserviceprofiler.ms_service_profiler_ext.exporters.exporter_summary.is_invaild_rid')
    def test_empty_rid(self, mock_is_invaild_rid):
        mock_is_invaild_rid.return_value = False
        all_data_df = pd.DataFrame({
            'rid': [''],
            'name': ['httpReq'],
            'start_time': [100],
            'end_time': [200]
        })
        result = get_new_total_time(all_data_df)
        self.assertEqual(result, {})

    @patch('msserviceprofiler.ms_service_profiler_ext.exporters.exporter_summary.is_invaild_rid')
    def test_invalid_rid(self, mock_is_invaild_rid):
        mock_is_invaild_rid.return_value = True
        all_data_df = pd.DataFrame({
            'rid': ['123'],
            'name': ['httpReq'],
            'start_time': [100],
            'end_time': [200]
        })
        result = get_new_total_time(all_data_df)
        self.assertEqual(result, {})

    def test_no_httpreq(self):
        all_data_df = pd.DataFrame({
            'rid': ['123'],
            'name': ['httpRes'],
            'start_time': [100],
            'end_time': [200]
        })
        result = get_new_total_time(all_data_df)
        self.assertEqual(result, {})

    def test_no_httpres(self):
        all_data_df = pd.DataFrame({
            'rid': ['123'],
            'name': ['httpReq'],
            'start_time': [100],
            'end_time': [200]
        })
        result = get_new_total_time(all_data_df)
        self.assertEqual(result, {})

    def test_valid_data(self):
        all_data_df = pd.DataFrame({
            'rid': ['123', '123'],
            'name': ['httpReq', 'httpRes'],
            'start_time': [100, 150],
            'end_time': [200, 250]
        })
        result = get_new_total_time(all_data_df)
        self.assertEqual(result, {RequestCSVFields.TOTAL_TIME: calculate_statistics([0.15])})


class TestGetNewTtftWaitTime(unittest.TestCase):
    
    def test_empty_data(self):
        # 测试当数据为空时的情况
        result = get_new_ttft_wait_time({})
        self.assertEqual(result, {})
        
    def test_empty_ttft_df(self):
        # 测试当ttft_df为空时的情况
        data = {'req_que_wait_df': pd.DataFrame({'que_wait_time': [1000, 2000]})}
        result = get_new_ttft_wait_time(data)
        self.assertEqual(result, {})
        
    def test_empty_que_wait_df(self):
        # 测试当que_wait_df为空时的情况
        data = {'req_ttft_df': pd.DataFrame({'ttft': [1000, 2000]})}
        result = get_new_ttft_wait_time(data)
        self.assertEqual(result, {})
        
    def test_valid_data(self):
        # 测试当数据有效时的情况
        data = {
            'req_ttft_df': pd.DataFrame({'ttft': [1000, 2000]}),
            'req_que_wait_df': pd.DataFrame({'que_wait_time': [1000, 2000]})
        }
        result = get_new_ttft_wait_time(data)
        expected_result = {
            RequestCSVFields.FIRST_TOKEN_LATENCY: {
                'avg': 1.5,
                'max': 2,
                'min': 1,
                'p50': 1.5,
                'p90': 1.9,
                'p99': 1.99
            },
            RequestCSVFields.WAITING_TIME: {
                'avg': 1.5,
                'max': 2,
                'min': 1,
                'p50': 1.5,
                'p90': 1.9,
                'p99': 1.99
            }
        }
        self.assertEqual(result, expected_result)


class TestExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = MagicMock()
        cls.args.output_path = '/path/to/output'

    @patch('msserviceprofiler.ms_service_profiler_ext.exporters.exporter_summary.logger')
    def test_export_empty_data(self, mock_logger):
        # 测试当data为空时的情况
        ExporterSummary.initialize(self.args)
        result = ExporterSummary.export({})
        mock_logger.warning.assert_called_once_with("The data is empty, please check")
        self.assertIsNone(result)

    @patch('msserviceprofiler.ms_service_profiler_ext.exporters.exporter_summary.logger')
    def test_export_empty_df(self, mock_logger):
        # 测试当all_data_df为空时的情况
        ExporterSummary.initialize(self.args)
        result = ExporterSummary.export({'tx_data_df': None})
        mock_logger.warning.assert_called_once_with("The data is empty, please check")
        self.assertIsNone(result)