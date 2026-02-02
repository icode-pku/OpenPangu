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

from __future__ import print_function

import datetime
import os
import unittest
from unittest.mock import patch, MagicMock

import pytest
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch

from components.utils.file_utils import FileChecker
from msit_llm.compare.cmp_utils import BasicDataInfo
from msit_llm.common.constant import (TOKEN_ID, DATA_ID, GOLDEN_DATA_PATH, MY_DATA_PATH,
                                      CMP_FAIL_REASON, GOLDEN_DTYPE, GOLDEN_SHAPE,
                                      GOLDEN_MAX_VALUE, GOLDEN_MIN_VALUE,
                                      GOLDEN_MEAN_VALUE, MY_DTYPE, MY_SHAPE,
                                      MY_MAX_VALUE, MY_MIN_VALUE, MY_MEAN_VALUE,
                                      CSV_GOLDEN_HEADER, GLOBAL_HISTORY_AIT_DUMP_PATH_LIST)
from msit_llm.compare.cmp_utils import (fill_row_data, load_as_torch_tensor,
                                         set_tensor_basic_info_in_row_data, compare_data, 
                                         read_data, save_compare_result_to_csv, align_tensors,
                                         read_csv_statistics, read_bin_statictics, convert_dict_values_to_fp32,
                                         compare_data_statistics, fill_row_data_statistics,
                                         save_statistics_compare_result_to_csv, save_compare_result_to_xlsx)

ori_file_common_check = FileChecker.common_check


def mock_file_common_check():
    def common_check(self):
        pass
    setattr(FileChecker, 'common_check', common_check)


def recover_file_common_check():
    setattr(FileChecker, 'common_check', ori_file_common_check)


@pytest.fixture(scope='module', autouse=True)
def golden_data_path():
    golden_data_path = "msit_dump_20240101_000000/torch_tensors/npu0_11111/1"
    yield golden_data_path


@pytest.fixture(scope='module', autouse=True)
def my_data_path():
    my_data_path = "msit_dump_20240101_000000/tensors/0_22222/2"
    yield my_data_path


@pytest.fixture(scope='module', autouse=True)
def sub_path():
    sub_path = "3_Prefill_layer/0_Attention/3_SelfAttention"
    yield sub_path


@pytest.fixture(scope='module', autouse=True)
def unvalid_data_path1():
    unvalid_data_path1 = "tensor_dump/tensors/0_22222/2"
    yield unvalid_data_path1


@pytest.fixture(scope='module', autouse=True)
def unvalid_data_path2():
    unvalid_data_path2 = "msit_dump_20240101_000000/tensors/0_22222/not_int"
    yield unvalid_data_path2


def test_get_token_id_from_golden_data_path(golden_data_path, my_data_path):
    mock_file_common_check()
    data_info = BasicDataInfo(my_data_path, golden_data_path)
    recover_file_common_check()
    assert data_info.token_id == 1


def test_get_token_id_from_my_data_path(golden_data_path, my_data_path):
    mock_file_common_check()
    data_info = BasicDataInfo(golden_data_path, my_data_path)
    recover_file_common_check()
    assert data_info.token_id == 2


def test_get_token_id_from_golden_data_path_sub(golden_data_path, my_data_path, sub_path):
    my_data_path = os.path.join(my_data_path, sub_path)
    golden_data_path = os.path.join(golden_data_path, sub_path)
    mock_file_common_check()
    data_info = BasicDataInfo(my_data_path, golden_data_path)
    recover_file_common_check()
    assert data_info.token_id == 1


def test_get_token_id_from_my_data_path_sub(golden_data_path, my_data_path, sub_path):
    my_data_path = os.path.join(my_data_path, sub_path)
    golden_data_path = os.path.join(golden_data_path, sub_path)
    mock_file_common_check()
    data_info = BasicDataInfo(golden_data_path, my_data_path)
    recover_file_common_check()
    assert data_info.token_id == 2


def test_get_token_id_unvalid1(golden_data_path, unvalid_data_path1):
    mock_file_common_check()
    data_info = BasicDataInfo(golden_data_path, unvalid_data_path1)
    recover_file_common_check()
    assert data_info.token_id == 0


def test_get_token_id_unvalid2(golden_data_path, unvalid_data_path2):
    mock_file_common_check()
    data_info = BasicDataInfo(golden_data_path, unvalid_data_path2)
    recover_file_common_check()
    assert data_info.token_id == 0


def test_given_token_id_data_id(golden_data_path, my_data_path):
    mock_file_common_check()
    data_info = BasicDataInfo(golden_data_path, my_data_path, token_id=3, data_id=4)
    recover_file_common_check()
    assert data_info.token_id == 3
    assert data_info.data_id == 4


def test_given_data_id(golden_data_path, my_data_path):
    mock_file_common_check()
    data_info = BasicDataInfo(golden_data_path, my_data_path, data_id=4)
    recover_file_common_check()
    assert data_info.token_id == 2
    assert data_info.data_id == 4


def test_given_token_id(golden_data_path, my_data_path):
    BasicDataInfo.count_data_id = 0
    mock_file_common_check()
    data_info1 = BasicDataInfo(golden_data_path, my_data_path, token_id=3)
    assert data_info1.token_id == 3
    assert data_info1.data_id == 0

    data_info1 = BasicDataInfo(golden_data_path, my_data_path, token_id=4)
    recover_file_common_check()
    assert data_info1.token_id == 4
    assert data_info1.data_id == 1


class TestFillRowData(unittest.TestCase):
    @patch('msit_llm.common.log.logger')  
    @patch('msit_llm.compare.cmp_utils.read_data')  
    def test_fill_row_data_with_missing_golden_data_file(self, mock_read_data, mock_logger):
        # 设置模拟数据
        class MockBasicDataInfo:
            def __init__(self):
                self.golden_data_path = 'missing_golden_data.npy'
                self.my_data_path = 'my_data.npy'
            
            @staticmethod
            def to_dict():
                return {}
 
        data_info = MockBasicDataInfo()
        mock_read_data.side_effect = [None, torch.tensor(np.array([1, 2, 3], dtype=np.float32))]
 
        # 调用函数
        result = fill_row_data(data_info)
 
        # 验证
        self.assertIn(CMP_FAIL_REASON, result)
        self.assertEqual(result[CMP_FAIL_REASON], f"golden_data_path: {data_info.golden_data_path} is not a file.")
 
 
class TestLoadAsTorchTensor(unittest.TestCase):
    def test_load_as_torch_tensor_with_loaded_data(self):
        data = np.array([1, 2, 3], dtype=np.float32)
        result = load_as_torch_tensor(None, data)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
 
    def test_load_as_torch_tensor_with_unsupported_dtype(self):
        data = np.array([1, 2, 3], dtype=np.int64)  # 假设int64是不支持的
        BasicDataInfo.TORCH_UNSUPPORTED_D_TYPE_MAP = {np.int64: torch.float32}
        result = load_as_torch_tensor(None, data)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.int64)

 
class TestSetTensorBasicInfoInRowData(unittest.TestCase):
    def test_set_tensor_basic_info_in_row_data(self):
        golden_data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        my_data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = set_tensor_basic_info_in_row_data(golden_data, my_data)
 
        self.assertEqual(result[GOLDEN_DTYPE], str(torch.float32))
        self.assertEqual(result[GOLDEN_SHAPE], str([3]))
        self.assertEqual(result[GOLDEN_MAX_VALUE], 3.0)
        self.assertEqual(result[GOLDEN_MIN_VALUE], 1.0)
        self.assertEqual(result[GOLDEN_MEAN_VALUE], 2.0)
        self.assertEqual(result[MY_DTYPE], str(torch.float32))
        self.assertEqual(result[MY_SHAPE], str([3]))
        self.assertEqual(result[MY_MAX_VALUE], 3.0)
        self.assertEqual(result[MY_MIN_VALUE], 1.0)
        self.assertEqual(result[MY_MEAN_VALUE], 2.0)


# Mock logger
logger = MagicMock()
 
# Mock functions
def read_atb_data(data_path):
    # Mock function to return a tensor
    return torch.tensor([1, 2, 3])
 
def safe_torch_load(data_path, map_location=torch.device("cpu")):
    # Mock function to return a tensor
    return torch.tensor([4, 5, 6])
 
class TestFunctions(unittest.TestCase):
    
    def test_save_compare_result_to_csv(self):
        # Mock datetime
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.datetime(2023, 10, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
            
            # Prepare data
            gathered_row_data = [
                {'Column1': 'Test1', 'Column2': 'Result1', 'Golden_Dtype': 'torch.float32', 'My_Dtype': 'torch.float32'},
                {'Column1': 'Test2', 'Column2': 'Result2', 'Golden_Dtype': 'torch.int8', 'My_Dtype': 'torch.float32'}
            ]
            output_path = "./test_output"
            
            # Mock os and pandas
            with patch('os.makedirs') as mock_makedirs, \
                 patch('os.path.join', return_value="mocked_path"), \
                 patch('pandas.DataFrame.to_csv') as mock_to_csv:
                
                # Call the function
                result_path = save_compare_result_to_csv(gathered_row_data, output_path)
                
                # Assertions
                self.assertEqual(result_path, "mocked_path")
                mock_makedirs.assert_called_once_with(output_path, exist_ok=True)
                self.assertEqual(len(gathered_row_data), 2) 


class TestAlignTensors(unittest.TestCase):
    def test_align_tensors_same_shape(self):
        """Test when tensors already have the same shape"""
        tensor1 = torch.randn(3, 4)
        tensor2 = torch.randn(3, 4)
        result1, result2 = align_tensors(tensor1, tensor2)
        self.assertEqual(result1.shape, tensor1.shape)
        self.assertEqual(result2.shape, tensor2.shape)

    def test_align_tensors_dim0(self):
        """Test alignment along dimension 0"""
        tensor1 = torch.randn(6, 4)  # Larger in dim0
        tensor2 = torch.randn(3, 4)  # Smaller in dim0
        result1, result2 = align_tensors(tensor1, tensor2)
        self.assertEqual(result1.shape, (6, 4))
        self.assertEqual(result2.shape, (6, 4))
    
    def test_align_tensors_dim1(self):
        """Test alignment along dimension 1"""
        tensor1 = torch.randn(3, 8)  # Larger in dim1
        tensor2 = torch.randn(3, 4)  # Smaller in dim1
        result1, result2 = align_tensors(tensor1, tensor2, dim=1)
        self.assertEqual(result1.shape, (3, 8))
        self.assertEqual(result2.shape, (3, 8))

    def test_align_tensors_multiple_dims(self):
        """Test alignment with higher dimensional tensors"""
        tensor1 = torch.randn(2, 6, 3)  # Larger in dim1
        tensor2 = torch.randn(2, 2, 3)  # Smaller in dim1
        result1, result2 = align_tensors(tensor1, tensor2, dim=1)
        self.assertEqual(result1.shape, (2, 6, 3))
        self.assertEqual(result2.shape, (2, 6, 3))

    def test_align_tensors_non_integer_multiple(self):
        """Test when tensors cannot be aligned due to non-integer multiple"""
        tensor1 = torch.randn(5, 4)  # Not divisible by 3
        tensor2 = torch.randn(3, 4)
        with self.assertRaises(ValueError):
            align_tensors(tensor1, tensor2)


class TestReadCSVStatistics(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up the temporary directory
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)
    
    def test_normal_case_with_output_row(self):
        """Test normal case with a valid Output row"""
        test_data = """InputOutput,Data1,Data2
Input,value1,value2
Output,value3,value4
Input,value5,value6"""
        
        test_file = os.path.join(self.test_dir, "test.csv")
        with open(test_file, 'w') as f:
            f.write(test_data)
        print(test_data)   
        expected = {'InputOutput': 'Output', 'Data1': 'value3', 'Data2': 'value4'}
        result = read_csv_statistics(test_file)
        self.assertEqual(result, expected)
    

class TestReadBinStatistics(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up the temporary directory
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)

    def test_read_complete_file(self):
        """Test reading a complete file with all required fields and end marker"""
        test_file = os.path.join(self.test_dir, "complete.bin")
        content = """dims=1,2,3
max=10.5
min=0.1
mean=5.0
l2norm=7.2
$End=1
"""
        with open(test_file, 'w') as f:
            f.write(content)
            
        result = read_bin_statictics(test_file)
        expected = {
            'dims': '1,2,3',
            'max': '10.5',
            'min': '0.1',
            'mean': '5.0',
            'l2norm': '7.2'
        }
        self.assertEqual(result, expected)


class TestConvertDictValuesToFP32(unittest.TestCase):
    def setUp(self):
        self.key_list = ['value', "value1"]
        self.sample_dict = {
            'value': 42,
            'value1': 'not_a_number' 
        }

    def test_normal_conversion(self):
        """Test normal conversion of valid values to float32"""
        result = convert_dict_values_to_fp32(self.key_list, self.sample_dict)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result['value'], np.float32)


        
class TestFillRowDataStatistics(unittest.TestCase):
    def setUp(self):
        # Create a basic data info mock
        self.data_info = MagicMock()
        self.data_info.golden_data_path = "/path/to/golden"
        self.data_info.my_data_path = "/path/to/my"
        self.data_info.to_dict.return_value = {
            TOKEN_ID: "test_token",
            DATA_ID: "test_data"
        }

    def test_my_file_not_exists(self):
        """Test when my file doesn't exist"""
        with patch('os.path.isfile', side_effect=[True, False]):
            result = fill_row_data_statistics(self.data_info)
            self.assertEqual(result[CMP_FAIL_REASON], f"my_data_path: {self.data_info.my_data_path} is not a file.")

    def test_both_files_not_exist(self):
        """Test when both files don't exist"""
        with patch('os.path.isfile', return_value=False):
            result = fill_row_data_statistics(self.data_info)
            self.assertEqual(result[CMP_FAIL_REASON], f"golden_data_path: {self.data_info.golden_data_path} is not a file.")


class TestSaveStatisticsCompareResultToCsv(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.sample_data = [
            {'token_id': '1', 'data_id': 'A', 'cmp_fail_reason': None},
            {'token_id': '2', 'data_id': 'B', 'cmp_fail_reason': 'Mismatch'},
            {'token_id': '3', 'data_id': 'C', 'cmp_fail_reason': None}
        ]
        self.columns = ['token_id', 'data_id', 'cmp_fail_reason']

    def tearDown(self):
        # Clean up the temporary directory
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)

    def test_successful_save_with_filtering(self):
        """Test that the function correctly saves data and filters out failed comparisons."""
        output_path = save_statistics_compare_result_to_csv(
            self.sample_data, 
            output_path=self.test_dir,
            columns=self.columns
        )
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 2)  # Should filter out the failed comparison
        self.assertListEqual(df['token_id'].tolist(), [1, 3]) 

    
class TestSaveCompareResultToXlsx(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.columns = ["token_id", "data_id", "golden_dtype", "my_dtype", "cmp_fail_reason"]
        
        # Sample data for testing
        self.sample_data = [
            {
                "token_id": "token1",
                "data_id": "data1",
                "golden_dtype": "torch.float32",
                "my_dtype": "torch.float32",
                "cmp_fail_reason": ""
            },
            {
                "token_id": "token2",
                "data_id": "data2",
                "golden_dtype": "torch.int8",
                "my_dtype": "torch.float32",
                "cmp_fail_reason": ""
            },
            {
                "token_id": "token3",
                "data_id": "data3",
                "golden_dtype": "torch.float32",
                "my_dtype": "torch.float32",
                "cmp_fail_reason": "data shape doesn't match."
            }
        ]

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_save_compare_result_success(self):
        """Test successful saving of comparison results"""
        sheet_names = ["layer"]
        output_path = save_compare_result_to_xlsx([self.sample_data], sheet_names, self.test_dir, self.columns)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        df = pd.read_excel(output_path, sheet_name="layer")
        self.assertEqual(len(df), 3)  # Should filter out int8 mismatch and shape mismatch

    def test_directory_creation_failure(self):
        """Test handling of directory creation failure"""
        with patch('os.makedirs', side_effect=OSError("Permission denied")):
            with self.assertRaises(OSError):
                save_compare_result_to_xlsx([self.sample_data], ["layer"], "/invalid/path", self.columns)