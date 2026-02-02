# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
import unittest
from unittest.mock import patch, mock_open, MagicMock

import torch

from msit_llm.compare.multi_block import get_cat_dim, get_multi_tensor_paths, multi_block_cmp


class TestGetCatDim(unittest.TestCase):
    def test_get_cat_dim_same_size(self):
        # Test two tensors have the same dimensions
        atb_tensors = [torch.randn(2, 3, 4) for _ in range(2)]
        torch_tensors = [torch.randn(2, 3, 4) for _ in range(2)]
        self.assertEqual(get_cat_dim(atb_tensors, torch_tensors), -1)

    def test_get_cat_dim_multiply_equal(self):
        atb_tensors = [torch.randn(2, 3, 4) for _ in range(2)]
        torch_tensors = [torch.randn(4, 3, 4) for _ in range(1)]  # 4 * 1 == 2 * 2
        self.assertEqual(get_cat_dim(atb_tensors, torch_tensors), 0)

    def test_get_cat_dim_no_match(self):
        atb_tensors = [torch.randn(2, 3, 5) for _ in range(2)]
        torch_tensors = [torch.randn(2, 4, 4) for _ in range(2)]
        self.assertEqual(get_cat_dim(atb_tensors, torch_tensors), -1)


class TestGetMultiTensorPaths(unittest.TestCase):

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.abspath', side_effect=lambda x: x)
    def test_file_exists(self, mock_abspath, mock_listdir, mock_exists):
        mock_listdir.return_value = ['0_npu_pid', '1_npu_pid']
        mock_exists.return_value = True
        read_data_mock = MagicMock(return_value='mocked tensor data')

        with patch('builtins.open', mock_open()), \
             patch('msit_llm.compare.multi_block.read_data', read_data_mock):
            result_path, result_tensors = get_multi_tensor_paths(
                '/base/data/path', '/node/path', 'output.pth'
            )
            self.assertEqual(result_path, os.path.join('/node/path', 'output.pth'))
            self.assertEqual(len(result_tensors), 2)
            self.assertTrue(all(tensor == 'mocked tensor data' for tensor in result_tensors))

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.abspath', side_effect=lambda x: x)
    def test_file_not_exists(self, mock_abspath, mock_listdir, mock_exists):
        # Test when file does not exist
        mock_listdir.return_value = ['0_npu_pid', '1_npu_pid']
        mock_exists.return_value = False

        with patch('msit_llm.compare.multi_block.read_data') as read_data_mock:
            result_path, result_tensors = get_multi_tensor_paths(
                '/base/data/path', '/node/path', 'output.pth'
            )
            self.assertIsNone(result_path)
            self.assertIsNone(result_tensors)
            read_data_mock.assert_not_called()


class TestMultiBlockCmp(unittest.TestCase):
    @patch('msit_llm.compare.multi_block.get_multi_tensor_paths')
    @patch('msit_llm.compare.multi_block.get_cat_dim')
    @patch('msit_llm.compare.multi_block.BasicDataInfo')
    @patch('msit_llm.compare.multi_block.fill_row_data')
    def test_multi_block_cmp(self, mock_fill_row_data, mock_basic_data_info, \
        mock_get_cat_dim, mock_get_multi_tensor_paths):
        # Create mock objects
        mock_atb_nodes = [
            MagicMock(op_type="LinearOperation", op_param={"hasBias": False}, tensor_path="/atb/tensor/path"),
            MagicMock(op_type="SomeOtherOp", tensor_path="/another/atb/path")
        ]
        mock_torch_nodes = [
            MagicMock(tensor_path="/torch/tensor/path"),
            MagicMock(tensor_path="/another/torch/path")
        ]
        mock_my_root_node = MagicMock()
        mock_next_sibling_node = MagicMock(
            op_type="ElewiseOperation",
            op_param={'elewiseType': 8},
            tensor_path="/next/sibling/path"
        )
        mock_my_root_node.get_next_sibling_node.return_value = mock_next_sibling_node
        mock_atb_tensor_path = "/base/atb/path"
        mock_torch_tensor_path = "/base/torch/path"
        mock_tensor_data = [torch.randn(2, 3), torch.randn(2, 3)] 
        mock_get_multi_tensor_paths.side_effect = [
            ('/first/atb/path', mock_tensor_data),
            ('/second/atb/path', mock_tensor_data),
            ('/first/torch/path', mock_tensor_data),
            ('/second/torch/path', mock_tensor_data)
        ]
        # Define a side effect for get_cat_dim that returns a finite sequence of values with a default.
        cat_dim_values = iter([0, -1])
        
        def cat_dim_side_effect(*args, **kwargs):
            try:
                return next(cat_dim_values)
            except StopIteration:
                return -1  # Default value if the iterator is exhausted.
        mock_get_cat_dim.side_effect = cat_dim_side_effect
        mock_basic_data_info.return_value = 'basic_data_info'
        mock_fill_row_data.return_value = 'filled_row_data'

        result = multi_block_cmp(mock_atb_nodes, mock_torch_nodes, mock_my_root_node, 
                                 mock_atb_tensor_path, mock_torch_tensor_path)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 'filled_row_data')
        self.assertEqual(result[1], 'filled_row_data')
        mock_my_root_node.get_next_sibling_node.assert_called_once_with(mock_atb_nodes[0])
        self.assertEqual(mock_get_multi_tensor_paths.call_count, 4)
        self.assertGreaterEqual(mock_get_cat_dim.call_count, 2)  # Ensure at least two calls.
        self.assertEqual(mock_basic_data_info.call_count, 2)
        self.assertEqual(mock_fill_row_data.call_count, 2)

