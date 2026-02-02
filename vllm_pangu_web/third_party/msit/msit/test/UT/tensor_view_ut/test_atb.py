# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import unittest

import torch

from components.tensor_view.ait_tensor_view.atb import read_atb_data, write_atb_data


class TestWriteReadAtbData(unittest.TestCase):
    def test_base_case(self):
        tensor = torch.rand(4, 4, 4, 4)

        write_atb_data(tensor, "./test.bin")
        actual_tensor = read_atb_data("./test.bin")
        os.remove("./test.bin")
        assert torch.equal(tensor, actual_tensor)

    def test_with_dtype(self):
        tensor = torch.rand(4, 4, 4, 4, 4, dtype=torch.float16)

        write_atb_data(tensor, "./test-dtype.bin")
        actual_tensor = read_atb_data("./test-dtype.bin")
        os.remove("./test-dtype.bin")
        assert torch.equal(tensor, actual_tensor)

    def test_with_dims(self):
        tensor = torch.rand(4, 4, 4, 4, 4, dtype=torch.float16)
        tensor.reshape(16, 16, 4, 1)

        write_atb_data(tensor, "./test-dims.bin")
        actual_tensor = read_atb_data("./test-dims.bin")
        os.remove("./test-dims.bin")
        assert torch.equal(tensor, actual_tensor)
