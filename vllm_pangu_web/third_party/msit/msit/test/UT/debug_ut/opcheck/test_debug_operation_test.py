import os
import unittest
import argparse
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from msit_opcheck.operation_test import OperationTest
from msit_opcheck.utils import NAMEDTUPLE_PRECISION_MODE


# mock掉OperationTest的父类
class MockTestCase:
    def __init__(self, methodName='debug_opcheck'):
        pass

OperationTest.__bases__ = (MockTestCase,)


class TestOperationTest(unittest.TestCase):

    def setUp(self):
        """Setup test case with default parameters"""
        self.case_info = {
            'op_type': ['test_op'],
            'op_name': 'test_operation',
            'op_param': {'input': [], 'output': []},
            'base_path': '/test/path',
            'data_path_dict': {},
            'precision_mode': 'float32',
            'precision_metric': ['abs', 'cos_sim', 'kl']
        }
        self.op_test = OperationTest(case_info=self.case_info)

    def test_validate_param_single_existing_param(self):
        result = self.op_test.validate_param('input')
        self.assertTrue(result)

    def test_validate_param_single_missing_param(self):
        with patch('components.debug.common.logger.error') as mock_logger:
            result = self.op_test.validate_param('nonexistent_param')
            self.assertFalse(result)
            mock_logger.assert_called_once()

    def test_valid_int_without_range(self):
        with self.assertRaises(argparse.ArgumentTypeError) as context:
            self.op_test.validate_int_range('11', range(1, 10), 'test_param')
            self.assertIn("[test_param]11 is not in range range(1, 10)!", str(context.exception))

    def test_validate_int_range_when_invalid_conversion_to_int(self):
        with self.assertRaises(RuntimeError) as context:
            self.op_test.validate_int_range('not_an_int', range(1, 10), 'test_param')
        self.assertIn("cannot be converted to an integer", str(context.exception))

    def test_validate_path_with_invalid_path(self):
        with self.assertRaises(RuntimeError) as context:
            self.op_test.validate_path(None)
            self.assertIn("not valid", str(context.exception))
    
    def test_force_dtype_when_fp32_conversion(self):
        input_tensors = [
            torch.tensor([1.0, 2.0], dtype=torch.float16),
            torch.tensor([3, 4], dtype=torch.uint8),
            torch.tensor([5.0, 6.0], dtype=torch.bfloat16)
        ]
        result = self.op_test.force_dtype(input_tensors, NAMEDTUPLE_PRECISION_MODE.force_fp32)
        self.assertEqual(result[0].dtype, torch.float32)
        self.assertEqual(result[1].dtype, torch.uint8)  # 不在float_types中，保持原有精度
        self.assertEqual(result[2].dtype, torch.float32)
        result = self.op_test.force_dtype(input_tensors, NAMEDTUPLE_PRECISION_MODE.force_fp16)
        self.assertEqual(result[0].dtype, torch.float16)
        self.assertEqual(result[2].dtype, torch.float16)

    @patch("msit_opcheck.operation_test.load_file_to_read_common_check")
    @patch("msit_opcheck.operation_test.OperationTest.validate_path")
    def test_setUp_with_valid_paths(self, mock_validate_path, mock_load_file_to_read_common_check):
        temp_dir = tempfile.TemporaryDirectory()
        base_path = temp_dir.name
        # Create sample input/output files
        input_files = [os.path.join(base_path, "input_0.npy"), os.path.join(base_path, "input_1.npy")]
        output_files = [os.path.join(base_path, "output_0.npy")]
        for file in input_files + output_files:
            array = np.random.rand(10).astype(np.float32)
            np.save(file, array)
        self.op_test.data_path_dict = {"input": input_files, "output": output_files}
        self.op_test.base_path = base_path
        mock_load_file_to_read_common_check.side_effect = lambda x: x  # mock掉校验函数不做任何事情

        self.op_test.setUp()
        self.assertEqual(len(self.op_test.in_tensors), 2)
        self.assertEqual(len(self.op_test.out_tensors), 1)

        temp_dir.cleanup()

    def test_teardown_when_case_info_not_equal_to_pass(self):
        self.case_info['excuted_information'] = 'SOME_ERROR'
        self.op_test.tearDown()
        self.assertEqual(self.case_info['excuted_information'], 'FAILED')

    def test_tensor_format_transform_when_nc1hwc0_to_nchw(self):
        # set case_info
        self.case_info['op_param']['input'] = [{
            'layout': 'NC1HWC0',
            'attr': [
                {'key': 'origin_format', 'value': {'s': 'NCHW'}},
                {'key': 'origin_shape', 'value': {'list': {'i': [1, 32, 32, 3]}}}
            ]
        }]

        input_tensor = torch.randn(1, 2, 32, 3, 16).numpy()
        transformed = self.op_test.tensor_format_transform(input_tensor, 'input', 0)
        self.assertEqual(transformed.shape, (1, 32, 32, 3))

    @patch("msit_opcheck.operation_test.OperationTest.tensor_format_transform")
    def test_execute_when_call_golden_calc_catch_error(self, mock_tensor_format_transform):
        self.case_info['op_param'] = {
            'input_desc': [{'shape': [2, 2], 'dtype': 'float32'}]
        }
        test_op = OperationTest(case_info=self.case_info)
        test_op.in_tensors = [torch.randn(2, 2)]
        mock_tensor_format_transform.side_effect = lambda x, y, z: x
        # 测试ZeroDivisionError
        test_op.golden_calc = MagicMock(side_effect=ZeroDivisionError("division by zero"))
        with self.assertRaises(RuntimeError):
            test_op.execute()
            self.assertEqual(self.case_info['fail_reason'], "ZeroDivisionError when calc golden")
        # 测试IndexError
        test_op.golden_calc = MagicMock(side_effect=IndexError("index out of range"))
        with self.assertRaises(RuntimeError):
            test_op.execute()
            self.assertEqual(self.case_info['fail_reason'], "IndexError when calc golden")
        # 测试其他error
        test_op.golden_calc = MagicMock(side_effect=MemoryError("OOM"))
        with self.assertRaises(RuntimeError):
            test_op.execute()
            self.assertEqual(self.case_info['fail_reason'], "Unexpected Error when calc golden")

    @patch("msit_opcheck.operation_test.OperationTest._OperationTest__golden_compare_all")
    @patch("msit_opcheck.operation_test.OperationTest.tensor_format_transform")
    def test_execute_when_input_and_output_is_normal(self, mock_tensor_format_transform, mock_golden_compare_all):
        self.case_info['op_param'] = {
            'input_desc': [{'shape': [2, 2], 'dtype': 'float32'}],
            'output_desc': [{'shape': [2, 2], 'dtype': 'float32'}]
        }
        test_op = OperationTest(case_info=self.case_info)
        test_op.in_tensors = [torch.randn(2, 2)]
        test_op.out_tensors = [torch.randn(2, 2)]
        test_op.golden_calc = lambda x: x
        mock_tensor_format_transform.side_effect = lambda x, y, z: x
        test_op.execute()

    def test_get_abs_pass_rate_when_all_pass(self):
        out = np.array([1.0, 2.0, 3.0])
        golden = np.array([1.0, 2.0, 3.0])
        etol = 0.001
        pass_rate, max_error = self.op_test.get_abs_pass_rate(out, golden, etol)
        self.assertEqual(pass_rate, 100.0)
        self.assertEqual(max_error, 0.0)

    def test_get_other_precisions_when_all_metrics_is_normal(self):
        out_tensors = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        golden_tensors = np.array([1.00001, 2.00001, 3.00001], dtype=np.float32)
        etol = 0.001
        result, message = self.op_test.get_other_precisions(out_tensors, golden_tensors, etol)
        self.assertEqual(len(result), 4)
        self.assertNotEqual(result[0], 'NaN')  # abs_pass_rate
        self.assertNotEqual(result[1], 'NaN')  # max_abs_error
        self.assertNotEqual(result[2], 'NaN')  # cos_sim
        self.assertNotEqual(result[3], 'NaN')  # kl_div
        self.assertEqual(message, "")

    @patch("msit_opcheck.operation_test.OperationTest.get_other_precisions")
    def test___golden_compare_all_successful_comparison_float32(self, mock_get_other_precisions):
        mock_get_other_precisions.side_effect = lambda a,b,c: ((1,1,1,1), "success")
        out_tensors = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        golden_tensors = [np.array([1.00001, 2.00001, 3.00001], dtype=np.float32)]
        
        self.op_test._OperationTest__golden_compare_all(out_tensors, golden_tensors)
        
        self.assertEqual(self.case_info['excuted_information'], 'PASS')
        self.assertEqual(len(self.case_info['res_detail']), 1)

    def test___golden_compare_all_size_mismatch(self):
        out_tensors = [np.array([1, 2, 3], dtype=np.float32)]
        golden_tensors = [np.array([1, 2], dtype=np.float32)]
        
        with self.assertRaises(RuntimeError):
            self.op_test._OperationTest__golden_compare_all(out_tensors, golden_tensors)
        assert "size of" in self.op_test.case_info['fail_reason']
