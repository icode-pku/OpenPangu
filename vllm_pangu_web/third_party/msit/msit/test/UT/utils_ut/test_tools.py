# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from collections import namedtuple

import numpy as np
import torch

from components.utils.tool import get_bin_data_from_dir, read_bin_data, load_file_to_read_common_check, \
    convert_bin_data_to_pt, convert_bin_data_to_npy, DEFAULT_PARSE_DTYPE, save_torch_data, save_npy_data, \
        load_file_to_read_common_check


class TestReadAndConvertBinData(unittest.TestCase):

    def setUp(self):
        # Create a temporary bin file 
        self.bin_data = np.random.randint(0, 256, size=100, dtype=np.uint8)
        self.bin_data_path = 'test_data.bin'
        with open(self.bin_data_path, 'wb') as f:
            f.write(self.bin_data.tobytes())

    def tearDown(self):
        # Delete temporary files
        if os.path.exists(self.bin_data_path):
            os.remove(self.bin_data_path)

    @patch('components.utils.tool.load_file_to_read_common_check')
    @patch('components.utils.tool.TensorBinFile')
    def test_read_bin_data_success(self, mock_tensor_bin_file, mock_load_file):
        real_bin_data_path = self.bin_data_path
        mock_load_file.return_value = real_bin_data_path
        mock_tensor_bin_file_instance = MagicMock()
        mock_tensor_bin_file.return_value = mock_tensor_bin_file_instance
        result = read_bin_data(self.bin_data_path)
        mock_load_file.assert_called_once_with(self.bin_data_path)
        mock_tensor_bin_file.assert_called_once_with(real_bin_data_path)
        self.assertEqual(result, mock_tensor_bin_file_instance)

    def test_read_bin_data_invalid_extension(self):
        bin_data_path = 'test_data.txt'
        with self.assertRaises(ValueError) as context:
            read_bin_data(bin_data_path)
        self.assertIn("must be end with .bin", str(context.exception))

    @patch('components.utils.tool.TensorBinFile')
    def test_convert_bin_data_to_pt_success(self, mock_tensor_bin_file):
        mock_tensor_bin_file_instance = MagicMock()
        mock_tensor_bin_file_instance.get_data.return_value = self.bin_data
        mock_tensor_bin_file.return_value = mock_tensor_bin_file_instance
        result = convert_bin_data_to_pt(mock_tensor_bin_file_instance)
        result = torch.tensor(result, dtype=torch.uint8)
        expected_result = torch.tensor(self.bin_data, dtype=torch.uint8)
        self.assertTrue(torch.equal(result, expected_result))

    @patch('components.utils.tool.TensorBinFile')
    def test_convert_bin_data_to_pt_empty_data(self, mock_tensor_bin_file):
        mock_tensor_bin_file_instance = MagicMock()
        mock_tensor_bin_file_instance.get_data.return_value = np.array([], dtype=np.uint8)
        mock_tensor_bin_file.return_value = mock_tensor_bin_file_instance
        result = convert_bin_data_to_pt(mock_tensor_bin_file_instance)
        result = torch.tensor(result, dtype=torch.uint8)
        expected_result = torch.tensor([], dtype=torch.uint8)
        self.assertTrue(torch.equal(result, expected_result))


Data = namedtuple('Data', ['data'])


class BinDumpData:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data


class TestConvertBinDataToNpy(unittest.TestCase):

    def setUp(self):
        self.input_data = [b'\x01\x02\x03', b'\x04\x05\x06']
        self.output_data = [b'\x07\x08\x09', b'\x0a\x0b\x0c']
        self.mock_input_data = [Data(data) for data in self.input_data]
        self.mock_output_data = [Data(data) for data in self.output_data]
        self.bin_dump_data = BinDumpData(self.mock_input_data, self.mock_output_data)

    def test_convert_bin_data_to_npy_default_dtype(self):
        inputs, outputs = convert_bin_data_to_npy(self.bin_dump_data)
        expected_inputs = [np.frombuffer(data, dtype=DEFAULT_PARSE_DTYPE) for data in self.input_data]
        expected_outputs = [np.frombuffer(data, dtype=DEFAULT_PARSE_DTYPE) for data in self.output_data]
        for i, (input_array, expected_input) in enumerate(zip(inputs, expected_inputs)):
            self.assertTrue(np.array_equal(input_array, expected_input), f"Input array {i} does not match expected value with default dtype")
        for i, (output_array, expected_output) in enumerate(zip(outputs, expected_outputs)):
            self.assertTrue(np.array_equal(output_array, expected_output), f"Output array {i} does not match expected value with default dtype")

    def test_convert_bin_data_to_npy_empty_input(self):
        empty_input_data = []
        bin_dump_data = BinDumpData(empty_input_data, self.mock_output_data)
        inputs, outputs = convert_bin_data_to_npy(bin_dump_data, dtype=np.uint8)
        self.assertEqual(inputs, [])
        self.assertNotEqual(outputs, [])


class TestSaveData(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory using tempfile
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # The TemporaryDirectory is automatically cleaned up when the context ends,
        # but we can explicitly close it here to ensure immediate cleanup.
        self.temp_dir.cleanup()

    def test_save_torch_data_success(self):
        pt_file_path = os.path.join(self.temp_dir.name, 'test.pt')
        pt_data = torch.tensor([1, 2, 3])
        save_torch_data(pt_data, pt_file_path)
        # # Verify file
        self.assertTrue(os.path.exists(pt_file_path))
        loaded_data = torch.load(pt_file_path)
        self.assertTrue(torch.equal(loaded_data, pt_data))

    def test_save_torch_data_invalid_extension(self):
        # Make sure to use self.temp_dir.name for the path
        pt_file_path = os.path.join(self.temp_dir.name, 'test.txt')  # corrected line
        with self.assertRaises(ValueError):  # Assuming that an invalid extension raises ValueError
            save_torch_data(torch.tensor([1, 2, 3]), pt_file_path)

    def test_save_torch_data_directory_creation(self):
        new_dir = os.path.join(self.temp_dir.name, 'new_subdir')
        pt_file_path = os.path.join(new_dir, 'test.pt')
        pt_data = torch.tensor([1, 2, 3])
        save_torch_data(pt_data, pt_file_path)
        # Verify file
        self.assertTrue(os.path.exists(pt_file_path))

    def test_save_npy_data_success(self):
        npy_file_path = os.path.join(self.temp_dir.name, 'test.npy')
        npy_data = np.array([1, 2, 3])
        save_npy_data(npy_file_path, npy_data)
        self.assertTrue(os.path.exists(npy_file_path))

        loaded_data = np.load(npy_file_path)
        self.assertTrue(np.array_equal(loaded_data, npy_data))

    def test_save_npy_data_invalid_extension(self):
        npy_file_path = os.path.join(self.temp_dir.name, 'test.txt')
        npy_data = np.array([1, 2, 3])
        #  ValueError
        with self.assertRaises(ValueError):
            save_npy_data(npy_file_path, npy_data)

    def test_save_npy_data_directory_creation(self):
        new_dir = os.path.join(self.temp_dir.name, 'new_subdir')
        npy_file_path = os.path.join(new_dir, 'test.npy')
        npy_data = np.array([1, 2, 3])
        save_npy_data(npy_file_path, npy_data)
        # Verify file
        self.assertTrue(os.path.exists(npy_file_path))


class TestGetBinDataFromDir(unittest.TestCase):

    def setUp(self):
        self.temp_dir = 'temp_test_dir'
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        sub_dir1 = os.path.join(self.temp_dir, 'subdir1')
        sub_dir2 = os.path.join(sub_dir1, 'subdir2')
        os.makedirs(sub_dir2)
        with open(os.path.join(self.temp_dir, 'file1.bin'), 'w') as f:
            f.write('test data')
        with open(os.path.join(sub_dir1, 'file2.bin'), 'w') as f:
            f.write('test data')
        with open(os.path.join(sub_dir2, 'file3.bin'), 'w') as f:
            f.write('test data')
        deep_dir = self.temp_dir
        for i in range(1, 3):
            deep_dir = os.path.join(deep_dir, f'deep{i}')
            os.makedirs(deep_dir)
            with open(os.path.join(deep_dir, f'file{i}.bin'), 'w') as f:
                f.write('test data')

    def tearDown(self):
        # Delete temporary directory
        if os.path.exists(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.temp_dir)

    @patch('components.utils.tool.load_file_to_read_common_check', side_effect=load_file_to_read_common_check)
    def test_get_bin_data_from_dir_success(self, mock_load_file):
        bin_files = get_bin_data_from_dir(self.temp_dir)
        expected_files = [
            os.path.realpath(os.path.join(self.temp_dir, 'file1.bin')),
            os.path.realpath(os.path.join(self.temp_dir, 'subdir1', 'file2.bin')),
            os.path.realpath(os.path.join(self.temp_dir, 'subdir1', 'subdir2', 'file3.bin')),
            os.path.realpath(os.path.join(self.temp_dir, 'deep1', 'file1.bin')),
            os.path.realpath(os.path.join(self.temp_dir, 'deep1', 'deep2', 'file2.bin'))
        ]
        self.assertCountEqual(bin_files, expected_files)

    @patch('components.utils.tool.load_file_to_read_common_check', side_effect=load_file_to_read_common_check)
    def test_get_bin_data_from_dir_no_bin_files(self, mock_load_file):
        for root, _, files in os.walk(self.temp_dir):
            for filename in files:
                if filename.endswith(".bin"):
                    os.remove(os.path.join(root, filename))
        bin_files = get_bin_data_from_dir(self.temp_dir)
        self.assertEqual(bin_files, [])