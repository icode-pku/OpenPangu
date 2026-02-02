import unittest
import os
import warnings
import tempfile
from unittest.mock import patch
from msit_llm.common.log import logger
from msit_llm.parameters_check.service_parameters_check import (
    extract_log_parameters,
    extract_txt_parameters,
    compare_service_parameters,
    service_params_check,
    generate_report,
    csv_input_safecheck
)


class TestUpdatedServiceParametersCheck(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_files = []
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")

    def tearDown(self):
        self.temp_dir.cleanup()
        for f in self.test_files:
            if os.path.exists(f):
                os.remove(f)

    @classmethod
    def tearDownClass(cls):
        output_dir = os.path.dirname(__file__)
        comparison_file_path = os.path.join(output_dir, "..", "..", "comparison_report.csv")
        os.remove(comparison_file_path)

    def create_temp_file(self, content, suffix=".log"):
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            dir=self.temp_dir.name,
            suffix=suffix,
            delete=False,
            encoding='utf-8'
        )
        temp_file.write(content)
        temp_file.close()
        self.test_files.append(temp_file.name)
        return temp_file.name

    def test_extract_log_parameters_order(self):
        """测试日志文件按行号顺序提取参数"""
        log_content = """
        [endpoint] Sampling parameters for request id: req1
        {
          "temperature": 0.7
        }
        Some other log
        [endpoint] Sampling parameters for request id: req2
        {
           "top_p": 0.9
        }
        """
        log_file = self.create_temp_file(log_content)
        result = extract_log_parameters(log_file)
        self.assertEqual(list(result.keys()), [1, 2])
        self.assertEqual(result[1]["temperature"], 0.7)
        self.assertEqual(result[2]["top_p"], 0.9)

    def test_extract_txt_multiple_requests(self):
        """测试文本文件多请求参数提取"""
        txt_content = """
        req1: SamplingParams(temperature=0.7)
        req2: SamplingParams(top_p=0.9)
        """
        txt_file = self.create_temp_file(txt_content, ".txt")
        result = extract_txt_parameters(txt_file)
        self.assertEqual(list(result.keys()), [1, 2])
        self.assertEqual(result[1]["temperature"], 0.7)
        self.assertEqual(result[2]["top_p"], 0.9)

    def test_compare_service_parameters_missing(self):
        """测试服务参数全缺失场景"""
        diffs = []
        params1 = {"temperature": 0.7}
        params2 = {"top_p": 0.9}
        compare_service_parameters(params1, params2, 1, diffs)

        # 验证存在的参数
        self.assertTrue(any(d["param"] == "temperature" and d["file2_value"] == "N/A" for d in diffs))
        self.assertTrue(any(d["param"] == "top_p" and d["file1_value"] == "N/A" for d in diffs))

    def test_service_cross_file_comparison(self):
        """测试跨文件类型比对（txt vs log）"""
        # 准备测试文件
        txt_content = "req0: SamplingParams(temperature=0.7, top_p=0.9)"
        log_content = """
        [endpoint] Sampling parameters for request id: any_id
        {
          "temperature": 0.7, 
          "top_p": 0.8
        }
        """

        txt_file = self.create_temp_file(txt_content, ".txt")
        log_file = self.create_temp_file(log_content, ".log")

        # 执行比对
        service_params_check(txt_file, log_file)

        # 验证报告
        report_path = "comparison_report.csv"
        with open(report_path) as f:
            content = f.read()
        self.assertIn("top_p,0.9,0.8", content)
        self.assertIn("req_order,param,file1_value,file2_value", content)

    def test_llm_cross_file_comparison(self):
        """测试跨文件类型比对（txt vs log）"""
        # 准备测试文件
        txt_content = "req0: SamplingParams(temperature=0.7, top_p=0.9, ignore_eos=False)do_sample:True"
        log_content = """
        [endpoint] Sampling parameters for request id: any_id
        {
           "temperature": [0.7], 
           "top_p": [0.8], 
           "do_sample": [False]
           "ignore_eos": [None]
        }
        """

        txt_file = self.create_temp_file(txt_content, ".txt")
        log_file = self.create_temp_file(log_content, ".log")

        # 执行比对
        service_params_check(txt_file, log_file)

        # 验证报告
        report_path = "comparison_report.csv"
        with open(report_path) as f:
            content = f.read()
        self.assertIn("top_p,0.9,0.8", content)
        self.assertIn("do_sample,True,False", content)
        self.assertIn("req_order,param,file1_value,file2_value", content)

    @patch("msit_llm.common.log.logger.warning")
    def test_no_sampling_params(self, mock_error):
        """测试日志中没有 SamplingParams 的情况"""
        # 准备测试文件
        txt_content = "req0: "
        log_content = """
        [endpoint] Sampling parameters for request id: any_id
        {
           "temperature": [0.7], 
        }
        """
        txt_file = self.create_temp_file(txt_content, ".txt")
        log_file = self.create_temp_file(log_content, ".log")
        # 执行比对
        service_params_check(txt_file, log_file)
        # 获取所有错误消息文本
        error_messages = [args[0] for args, _ in mock_error.call_args_list]
        # 验证必须包含的关键错误
        self.assertIn("Please check whether the SamplingParams(...) is correct", error_messages[0])

    @patch("msit_llm.common.log.logger.warning")
    def test_unclosed_parentheses(self, mock_error):
        """测试 SamplingParams 括号未闭合"""
        # 准备测试文件
        txt_content = "req0: SamplingParams("
        log_content = """
        [endpoint] Sampling parameters for request id: any_id
        {
           "temperature": [0.7], 
        }
        """
        txt_file = self.create_temp_file(txt_content, ".txt")
        log_file = self.create_temp_file(log_content, ".log")
        # 执行比对
        service_params_check(txt_file, log_file)
        # 获取所有错误消息文本
        error_messages = [args[0] for args, _ in mock_error.call_args_list]
        # 验证必须包含的关键错误
        self.assertIn("Please check whether the SamplingParams(...) is correct", error_messages[0])

    @patch("msit_llm.common.log.logger.warning")
    def test_invalid_parameter_format(self, mock_error):
        """测试参数格式错误（无等号）"""
        # 准备测试文件
        txt_content = "req0: SamplingParams(temperature0.7)"
        log_content = """
        [endpoint] Sampling parameters for request id: any_id
        {
           "temperature": [0.7], 
        }
        """
        txt_file = self.create_temp_file(txt_content, ".txt")
        log_file = self.create_temp_file(log_content, ".log")
        # 执行比对
        service_params_check(txt_file, log_file)
        error_messages = [args[0] for args, _ in mock_error.call_args_list]
        self.assertIn("Please check whether the parameter: temperature0.7 is correct.", error_messages[0])

    @patch("msit_llm.common.log.logger.warning")
    def test_invalid_do_sample(self, mock_warning):
        """测试 do_sample 解析异常"""
        # 准备测试文件
        txt_content = "req0: SamplingParams(temperature=0.7)do_sample:-"
        log_content = """
        [endpoint] Sampling parameters for request id: any_id
        {
           "temperature": [0.7], 
        }
        """
        txt_file = self.create_temp_file(txt_content, ".txt")
        log_file = self.create_temp_file(log_content, ".log")
        # 执行比对
        service_params_check(txt_file, log_file)

        mock_warning.assert_called_with("Please check that the content of the do_sample is correct.")

    @patch("msit_llm.common.log.logger.warning")
    def test_mismatched_brackets_params(self, mock_error):
        """测试参数括号不匹配"""
        # 准备测试文件
        txt_content = "req0: SamplingParams(temperature=1.0, stop=['stop1', 'stop2'})"
        log_content = """
        [endpoint] Sampling parameters for request id: any_id
        {
           "temperature": [0.7], 
        }
        """
        txt_file = self.create_temp_file(txt_content, ".txt")
        log_file = self.create_temp_file(log_content, ".log")
        # 执行比对
        service_params_check(txt_file, log_file)
        # 获取所有错误消息文本
        error_messages = [args[0] for args, _ in mock_error.call_args_list]
        # 验证必须包含的关键错误
        self.assertIn("Mismatched brackets: expected ']', got '}' in line: req0: SamplingParams("
                      "temperature=1.0, stop=['stop1', 'stop2'})", error_messages[0])

    def test_multi_request_report(self):
        """测试多请求报告生成"""
        # 生成测试差异数据
        diffs = [
            {"req_order": 1, "param": "temperature", "file1_value": 0.7, "file2_value": 0.8},
            {"req_order": 2, "param": "top_p", "file1_value": 0.9, "file2_value": 1.0}
        ]

        # 测试报告生成
        report_path = os.path.join(self.temp_dir.name, "multi_report.csv")
        generate_report(diffs, report_path)

        # 验证报告内容
        with open(report_path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)  # Header + 2 rows
        self.assertIn("1,temperature,0.7,0.8", lines[1])
        self.assertIn("2,top_p,0.9,1.0", lines[2])

    def test_none_handling_in_report(self):
        """测试None值在报告中的转换"""
        diffs = [{
            "req_order": 0,
            "param": "seed",
            "file1_value": None,
            "file2_value": 42
        }]

        report_path = os.path.join(self.temp_dir.name, "none_report.csv")
        generate_report(diffs, report_path)

        with open(report_path) as f:
            content = f.read()
        self.assertIn("0,seed,None,42", content)

    def test_mismatched_request_counts(self):
        """测试请求数量不一致场景"""
        # 准备测试文件
        log1_content = """
        [endpoint] Sampling parameters for request id: req1
        {
          "temperature": 0.7
        }
        [endpoint] Sampling parameters for request id: req2
        {
          "temperature": 0.8
        }
        """
        log2_content = """
        [endpoint] Sampling parameters for request id: req1
        {
          "temperature": 0.7
        }
        """

        log1 = self.create_temp_file(log1_content)
        log2 = self.create_temp_file(log2_content)

        # 执行比对并捕获异常
        with self.assertLogs(logger, level='ERROR') as cm:
            service_params_check(log1, log2)
        self.assertTrue(any("ERROR" in log for log in cm.output))
    
    def test_malicious_key_raises_exception(self):
        """测试恶意名称触发异常"""
        differences = [{'req_order': 1, 'param': '=;+', 'file1_value': 0.5, 'file2_value': 0.6}]
        with self.assertRaises(ValueError):
            csv_input_safecheck(differences)
    
    def test_ast_abnormal_value(self):
        """测试ast无法处理的字符串"""
        txt_content = "req0: SamplingParams(temperature=0.7, frequency_penalty=0.9)do_sample:True"
        log_content = """
                [endpoint] Sampling parameters for request id: any_id
                {
                  "temperature": 0.7, 
                  "frequency_penalty": null,
                  "watermark": true,
                  "do_sample": false,
                }
                """
        txt_file = self.create_temp_file(txt_content, ".txt")
        log_file = self.create_temp_file(log_content, ".log")
        # 执行比对
        service_params_check(txt_file, log_file)
        # 验证报告
        report_path = "comparison_report.csv"
        with open(report_path) as f:
            content = f.read()
        self.assertIn("do_sample,True,False", content)
        self.assertIn("watermark,N/A,True", content)
        self.assertIn("frequency_penalty,0.9,None", content)
        self.assertIn("req_order,param,file1_value,file2_value", content)

