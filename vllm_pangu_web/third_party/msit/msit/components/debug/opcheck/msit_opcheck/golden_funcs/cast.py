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

import torch
import numpy as np
import tensorflow as tf

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP

STR_TO_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bfloat16": torch.bfloat16,
    "bool": torch.bool
}


class CastOperation(OperationTest):
    def golden_calc(self, in_tensors):
        src_type = DATA_TYPE_MAP[self.op_param['input_desc'][0]['dtype']]
        dst_type = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        data = in_tensors[0]
        
        if src_type == "float32" and dst_type == "int64":
            out = torch.tensor(data, dtype=torch.int64).numpy()
            return [out]
        if dst_type == "complex32":
            _shape = list(data.shape)
            data = data.reshape(_shape + [1])
            imag = np.zeros(_shape + [1], dtype=np.float16)
            res = np.concatenate((data, imag), axis=-1)
            return [res]
        if src_type == "uint1":
            data = np.unpackbits(data)
        if src_type == "float32" and dst_type == "float16":
            out = torch.tensor(data, dtype=torch.float16).numpy()
            return [out]
        if dst_type == "complex64":
            out = tf.cast(data, dtype=dst_type)
            with tf.compat.v1.Session() as sess:
                res = sess.run(out)
            return [res]
            
        if src_type == "uint32":
            return [data.astype(dst_type)]
        data_tensor = self.numpy_to_torch_tensor(data)
        out_dtype_torch = self.str_to_dtype(dst_type)
        out = data_tensor.to(out_dtype_torch)
        if dst_type == "bfloat16":
            out = out.type(torch.float32)
            res = out.numpy().astype(tf.bfloat16.as_numpy_dtype)
        else:
            res = out.numpy()
        return [res]
    
    def test_cast(self):
        self.execute()

    def numpy_to_torch_tensor(self, np_array: np.ndarray, is_complex32: bool = False):
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
                    raise RuntimeError(f"Can only transfer np.float16 to torch.complex32. Not: {np_array.dtype.name}")
            else:
                return torch.from_numpy(np_array)

    def str_to_dtype(self, dtype_str):
        if dtype_str in STR_TO_DTYPE.keys():
            return STR_TO_DTYPE[dtype_str]
        else:
            raise ValueError('Unsupported dtype: {}'.format(dtype_str))

