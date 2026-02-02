# -*- coding: utf-8 -*-
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

import numpy
import torch


DATA_TYPE_MAP = {
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


def numpy_int4():
    try:
        from ml_dtypes import int4
        return int4
    except ImportError as e:
        raise RuntimeError("ml_dtypes is needed to support int4 dtype!!! "
                           "Please install with `pip3 install ml_dtypes`.") from e


def numpy_bfloat16():
    import tensorflow
    return tensorflow.bfloat16.as_numpy_dtype


def numpy_float8_e5m2():
    try:
        from ml_dtypes import float8_e5m2
        return float8_e5m2
    except ImportError as e:
        raise RuntimeError("ml_dtypes is needed to support float8_e5m2 dtype!!! "
                           "Please install with `pip3 install ml_dtypes`.") from e


def numpy_float8_e4m3fn():
    try:
        from ml_dtypes import float8_e4m3fn
        return float8_e4m3fn
    except ImportError as e:
        raise RuntimeError("ml_dtypes is needed to support float8_e4m3fn dtype!!! "
                           "Please install with `pip3 install ml_dtypes`.") from e


def numpy_float8_e8m0():
    try:
        from en_dtypes import float8_e8m0
        return float8_e8m0
    except ImportError as e:
        raise RuntimeError("en_dtypes is needed to support float8_e8m0 dtype!!! "
                           "Please install with `pip3 install en_dtypes`.") from e


def numpy_float4_e2m1():
    try:
        from en_dtypes import float4_e2m1
        return float4_e2m1
    except ImportError as e:
        raise RuntimeError("en_dtypes is needed to support float4_e2m1 dtype!!! "
                           "Please install with `pip3 install en_dtypes`.") from e


def numpy_float4_e1m2():
    try:
        from en_dtypes import float4_e1m2
        return float4_e1m2
    except ImportError as e:
        raise RuntimeError("en_dtypes is needed to support float4_e1m2 dtype!!! "
                           "Please install with `pip3 install en_dtypes`.") from e


DTYPE_CONVERSION_MAP = {
    "int4": numpy_int4,
    "bfloat16": numpy_bfloat16,
    "float8_e5m2": numpy_float8_e5m2,
    "float8_e4m3fn": numpy_float8_e4m3fn,
    "float8_e8m0": numpy_float8_e8m0,
    "float4_e2m1": numpy_float4_e2m1,
    "float4_e1m2": numpy_float4_e1m2
}


def bfloat16_conversion(container):
    """
    Convert bfloat16 string to numpy dtype
    """
    # noinspection PyUnresolvedReferences
    import tensorflow
    return [dtype if dtype != "bfloat16" else tensorflow.bfloat16.as_numpy_dtype for dtype in container]


def bfloat16_conversion_v2(container):
    """
    Convert bfloat16/int4/fp8 string to numpy dtype
    """
    ret = list(container)
    special_dtypes = ("bfloat16", "int4", "float8_e5m2", "float8_e4m3fn", "float8_e8m0", "float4_e2m1", "float4_e1m2")
    for sd in special_dtypes:
        for idx, dtype in enumerate(ret):
            if sd == dtype:
                ret[idx] = DTYPE_CONVERSION_MAP.get(sd)()
    return ret


def numpy_to_torch_tensor(np_array: numpy.ndarray, is_complex32=False):
    """
    Convert numpy data to torch.tensor
    """
    if "bfloat16" in str(np_array.dtype):
        if hasattr(torch, "frombuffer"):
            return torch.frombuffer(np_array, dtype=torch.bfloat16).reshape(np_array.shape)
        else:
            np_fp32 = np_array.astype("float32")
            t_fp32 = torch.from_numpy(np_fp32)
            return t_fp32.to(torch.bfloat16, copy=False)
    else:
        if is_complex32:
            if np_array.dtype.name == "float16":
                ret = torch.from_numpy(np_array)
                return ret.view(torch.complex32)
            else:
                raise RuntimeError(f"Can only transfer numpy.float16 to torch.complex32. Not: {np_array.dtype.name}")
        else:
            return torch.from_numpy(np_array)
