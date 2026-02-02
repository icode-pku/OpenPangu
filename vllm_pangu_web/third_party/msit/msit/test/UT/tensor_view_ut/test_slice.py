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

from components.tensor_view.ait_tensor_view.operation import SliceOperation


class TestSliceOperation(unittest.TestCase):
    def test_single_dimension(self):
        tensor = torch.rand(4, 4, 4)
        op1 = SliceOperation("[3]")
        op2 = SliceOperation("[:3]")
        op3 = SliceOperation("[1:2:]")
        op4 = SliceOperation("[::2]")
        assert torch.equal(op1.process(tensor), tensor[3])
        assert torch.equal(op2.process(tensor), tensor[:3])
        assert torch.equal(op3.process(tensor), tensor[1:2:])
        assert torch.equal(op4.process(tensor), tensor[::2])

    def test_negative(self):
        tensor = torch.rand(1, 2, 3)
        op1 = SliceOperation("[-1]")
        op2 = SliceOperation("[..., -1:-3]")
        op3 = SliceOperation("[0, -1:-2, -2:-3]")
        assert torch.equal(op1.process(tensor), tensor[-1])
        assert torch.equal(op2.process(tensor), tensor[..., -1:-3])
        assert torch.equal(op3.process(tensor), tensor[0, -1:-2, -2:-3])

    def test_ellipsis(self):
        tensor = torch.rand(3, 4, 5)
        op = SliceOperation("[..., 1]")
        assert torch.equal(op.process(tensor), tensor[..., 1])

    def test_multi_dimension(self):
        tensor = torch.rand(3, 4, 5)
        assert torch.equal(SliceOperation("[1, 2:4, 0]").process(tensor), tensor[1, 2:4, 0])

    def test_invalid_dimension(self):
        tensor = torch.rand(3, 4)
        with self.assertRaises(ValueError):
            op = SliceOperation("[1, 2, 3]")
            op.process(tensor)

    def test_index_out_of_range(self):
        tensor = torch.rand(3, 4)
        with self.assertRaises(IndexError):
            op = SliceOperation("[3, 0]")
            op.process(tensor)
