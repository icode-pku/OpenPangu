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

import shutil
import os
import tempfile
from unittest import TestCase
from unittest import mock

import numpy as np

from msit_llm.compare.cmp_weight import find_safetensors_files, dequant, compare_weight, logger


class TestFindSafetensorsFiles(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        # Ensure the temporary directory is deleted after all tests
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        # Create temporary safetensors files for testing
        self.test_files = [
            'model1.safetensors',
            'model2.safetensors',
            'other_file.txt'
        ]
        self.created_files = []
        for file_name in self.test_files:
            file_path = os.path.join(self.temp_dir, file_name)
            with open(file_path, 'w') as f:
                f.write('dummy content')
            self.created_files.append(file_path)

    def test_find_safetensors_files_with_safetensors(self):
        safetensors_files = find_safetensors_files(self.temp_dir)
        expected_files = [f for f in self.created_files if f.endswith('.safetensors')]
        self.assertEqual(sorted(safetensors_files), sorted(expected_files))


class TestDequant(TestCase):
    def test_dequant_with_scalar(self):
        weight = 5.0
        weight_offset = 2.0
        weight_scale = 3.0
        expected_result = (weight - weight_offset) * weight_scale
        result = dequant(weight, weight_offset, weight_scale)
        self.assertAlmostEqual(result, expected_result)

    def test_dequant_with_numpy_array(self):
        weight = np.array([1.0, 2.0, 3.0])
        weight_offset = np.array([0.5, 0.5, 0.5])
        weight_scale = np.array([2.0, 2.0, 2.0])
        expected_result = (weight - weight_offset) * weight_scale
        result = dequant(weight, weight_offset, weight_scale)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_dequant_with_broadcasting(self):
        weight = np.array([[1.0, 2.0], [3.0, 4.0]])
        weight_offset = np.array([0.5, 0.5])
        weight_scale = 2.0
        expected_result = (weight - weight_offset) * weight_scale
        result = dequant(weight, weight_offset, weight_scale)
        np.testing.assert_array_almost_equal(result, expected_result)


class TestCompareWeight(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        # Create temporary safetensors files for testing
        self.test_files = [
            'model1.safetensors',
            'model2.safetensors',
            'other_file.txt'
        ]
        self.created_files = []
        for file_name in self.test_files:
            file_path = os.path.join(self.temp_dir, file_name)
            with open(file_path, 'w') as f:
                f.write('dummy content')
            self.created_files.append(file_path)

    def tearDown(self):
        # Clean up any created files after each test
        for file_path in self.created_files:
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error("Error deleting file %s: %s", file_path, e)
    @mock.patch('msit_llm.compare.cmp_weight.find_safetensors_files', return_value=[])
    @mock.patch('msit_llm.compare.cmp_weight.logger.error')
    def test_compare_weight_no_files(self, mock_logger_error, mock_find_files):
        gp_path = os.path.join(self.temp_dir, 'gp')
        mp_path = os.path.join(self.temp_dir, 'mp')
        output_path = os.path.join(self.temp_dir, 'output.csv')

        with self.assertRaises(FileNotFoundError):
            compare_weight(gp_path, mp_path, output_path)
        mock_logger_error.assert_called_once_with("No .safetensors files found in the directory.")