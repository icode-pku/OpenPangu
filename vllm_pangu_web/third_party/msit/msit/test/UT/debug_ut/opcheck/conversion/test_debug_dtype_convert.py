# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
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

from unittest import TestCase
from unittest.mock import MagicMock

import en_dtypes
from en_dtypes import float8_e8m0, float4_e2m1, float4_e1m2
import ml_dtypes
from ml_dtypes import int4, float8_e5m2, float8_e4m3fn, bfloat16
import numpy as np
import torch

from msit_opcheck.conversion.dtype_convert import (
    DATA_TYPE_MAP,
    DTYPE_CONVERSION_MAP,
    numpy_int4,
    numpy_bfloat16,
    numpy_float8_e5m2,
    numpy_float8_e4m3fn,
    numpy_float8_e8m0,
    numpy_float4_e2m1,
    numpy_float4_e1m2,
    bfloat16_conversion_v2,
    numpy_to_torch_tensor
)


class TestDtypeConvert(TestCase):
    def test_data_type_map(self):
        target_map = {
            "DT_FLOAT": "float32",
            "DT_FLOAT16": "float16",
            "DT_DOUBLE": "float64",
            "DT_BOOL": "bool",
            "DT_INT8": "int8",
            "DT_INT16": "int16",
            "DT_INT32": "int32",
            "DT_INT64": "int64",
            "DT_UINT1": "uint1",
            "DT_UINT8": "uint8",
            "DT_UINT16": "uint16",
            "DT_UINT32": "uint32",
            "DT_UINT64": "uint64",
            "DT_COMPLEX32": "complex32",
            "DT_COMPLEX64": "complex64",
            "DT_COMPLEX128": "complex128"
        }
        self.assertEqual(DATA_TYPE_MAP, target_map)

    def test_dtype_conversion_map(self):
        target_map = {
            "int4": numpy_int4,
            "bfloat16": numpy_bfloat16,
            "float8_e5m2": numpy_float8_e5m2,
            "float8_e4m3fn": numpy_float8_e4m3fn,
            "float8_e8m0": numpy_float8_e8m0,
            "float4_e2m1": numpy_float4_e2m1,
            "float4_e1m2": numpy_float4_e1m2
        }
        self.assertEqual(DTYPE_CONVERSION_MAP, target_map)

    def test_numpy_int4(self):
        ori_int4 = getattr(ml_dtypes, 'int4')
        del ml_dtypes.int4
        with self.assertRaises(RuntimeError) as context:
            numpy_int4()
        self.assertEqual(str(context.exception), "ml_dtypes is needed to support int4 dtype!!! "
                         "Please install with `pip3 install ml_dtypes`.")

        setattr(ml_dtypes, 'int4', ori_int4)
        self.assertEqual(int4, numpy_int4())

    def test_numpy_float8_e5m2(self):
        ori_float8_e5m2 = getattr(ml_dtypes, 'float8_e5m2')
        del ml_dtypes.float8_e5m2
        with self.assertRaises(RuntimeError) as context:
            numpy_float8_e5m2()
        self.assertEqual(str(context.exception), "ml_dtypes is needed to support float8_e5m2 dtype!!! "
                         "Please install with `pip3 install ml_dtypes`.")

        setattr(ml_dtypes, 'float8_e5m2', ori_float8_e5m2)
        self.assertEqual(float8_e5m2, numpy_float8_e5m2())

    def test_numpy_float8_e4m3fn(self):
        ori_float8_e4m3fn = getattr(ml_dtypes, 'float8_e4m3fn')
        del ml_dtypes.float8_e4m3fn
        with self.assertRaises(RuntimeError) as context:
            numpy_float8_e4m3fn()
        self.assertEqual(str(context.exception), "ml_dtypes is needed to support float8_e4m3fn dtype!!! "
                         "Please install with `pip3 install ml_dtypes`.")

        setattr(ml_dtypes, 'float8_e4m3fn', ori_float8_e4m3fn)
        self.assertEqual(float8_e4m3fn, numpy_float8_e4m3fn())

    def test_numpy_float8_e8m0(self):
        ori_float8_e8m0 = getattr(en_dtypes, 'float8_e8m0')
        del en_dtypes.float8_e8m0
        with self.assertRaises(RuntimeError) as context:
            numpy_float8_e8m0()
        self.assertEqual(str(context.exception), "en_dtypes is needed to support float8_e8m0 dtype!!! "
                         "Please install with `pip3 install en_dtypes`.")

        setattr(en_dtypes, 'float8_e8m0', ori_float8_e8m0)
        self.assertEqual(float8_e8m0, numpy_float8_e8m0())

    def test_numpy_float4_e2m1(self):
        ori_float4_e2m1 = getattr(en_dtypes, 'float4_e2m1')
        del en_dtypes.float4_e2m1
        with self.assertRaises(RuntimeError) as context:
            numpy_float4_e2m1()
        self.assertEqual(str(context.exception), "en_dtypes is needed to support float4_e2m1 dtype!!! "
                         "Please install with `pip3 install en_dtypes`.")

        setattr(en_dtypes, 'float4_e2m1', ori_float4_e2m1)
        self.assertEqual(float4_e2m1, numpy_float4_e2m1())

    def test_numpy_float4_e1m2(self):
        ori_float4_e1m2 = getattr(en_dtypes, 'float4_e1m2')
        del en_dtypes.float4_e1m2
        with self.assertRaises(RuntimeError) as context:
            numpy_float4_e1m2()
        self.assertEqual(str(context.exception), "en_dtypes is needed to support float4_e1m2 dtype!!! "
                         "Please install with `pip3 install en_dtypes`.")

        setattr(en_dtypes, 'float4_e1m2', ori_float4_e1m2)
        self.assertEqual(float4_e1m2, numpy_float4_e1m2())

    def test_bfloat16_conversion_v2(self):
        container = (
            float, "int4", "float8_e5m2",
            "float8_e4m3fn", "float8_e8m0", "dtype",
            "float4_e2m1", "float4_e1m2"
        )
        target = [
            float, int4, float8_e5m2, float8_e4m3fn,
            float8_e8m0, "dtype", float4_e2m1, float4_e1m2
        ]
        ret = bfloat16_conversion_v2(container)
        self.assertEqual(ret, target)

    def test_numpy_to_torch_tensor(self):
        np_array = np.zeros((1,), dtype=bfloat16)
        target = torch.tensor([0], dtype=torch.bfloat16)
        if hasattr(torch, "frombuffer"):
            ret = numpy_to_torch_tensor(np_array)
            self.assertTrue((ret == target).all())

            ori_frombuffer = getattr(torch, 'frombuffer')
            del torch.frombuffer
            ret = numpy_to_torch_tensor(np_array)
            self.assertTrue((ret == target).all())
            setattr(torch, 'frombuffer', ori_frombuffer)
        else:
            ret = numpy_to_torch_tensor(np_array)
            self.assertTrue((ret == target).all())

            mock_frombuffer = MagicMock()
            mock_frombuffer.reshape.return_value = target
            setattr(torch, 'frombuffer', mock_frombuffer)
            ret = numpy_to_torch_tensor(np_array)
            mock_frombuffer.assert_called_with(np_array, dtype=torch.bfloat16)
            mock_frombuffer.reshape.assert_called_with((1,))
            self.assertTrue((ret == target).all())
            del torch.frombuffer

        np_array = np.zeros((1), dtype=np.float32)
        target = torch.tensor([0], dtype=torch.float32)
        ret = numpy_to_torch_tensor(np_array)
        self.assertTrue((ret == target).all())

        with self.assertRaises(RuntimeError) as context:
            numpy_to_torch_tensor(np_array, is_complex32=True)
        self.assertEqual(str(context.exception), 'Can only transfer numpy.float16 to torch.complex32. '
                         f'Not: {np_array.dtype.name}')

        np_array = np.zeros((2), dtype=np.float16)
        target = torch.tensor([0], dtype=torch.complex32)
        ret = numpy_to_torch_tensor(np_array, is_complex32=True)
        self.assertTrue((ret == target).all())
