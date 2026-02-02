# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch

from components.tensor_view.ait_tensor_view.operation import PermuteOperation


class TestPermuteOperation(unittest.TestCase):
    tensor = torch.rand(1, 2, 3, 4)

    def test_unequal_dimension(self):
        op = PermuteOperation("(3,4,2)")
        with self.assertRaises(ValueError):
            op.process(self.tensor)

    def test_duplicate(self):
        op = PermuteOperation("(2,3,2,4,0)")
        with self.assertRaises(ValueError):
            op.process(self.tensor)

    def test_not_n1(self):
        op = PermuteOperation("(0,1,2,3,5)")
        with self.assertRaises(ValueError):
            op.process(self.tensor)

    def test_valid(self):
        op = PermuteOperation("(3,1,2,0)")
        assert torch.equal(op.process(self.tensor), self.tensor.permute(3, 1, 2, 0))
