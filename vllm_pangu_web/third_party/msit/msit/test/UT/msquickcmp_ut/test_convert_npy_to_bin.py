# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import argparse
import unittest
import os
import numpy as np
from msquickcmp.common.convert import convert_npy_to_bin


class TestConvertNpyToBin(unittest.TestCase):
    def setUp(self):
        self.npy_path = 'convert_test.npy'
        self.bin_path = 'convert_test.bin'
        self.args = argparse.Namespace(input_path=self.npy_path)

    def tearDown(self):
        if os.path.exists(self.npy_path):
            os.remove(self.npy_path)
        if os.path.exists(self.bin_path):
            os.remove(self.bin_path)

    def test_convert_npy_to_bin(self):
        # create a test npy file
        npy_data = np.array([1, 2, 3])
        np.save(self.npy_path, npy_data)

        # call the function to convert npy to bin
        convert_npy_to_bin(self.args.input_path)

        # check if the bin file is generated
        assert os.path.exists(self.bin_path)



