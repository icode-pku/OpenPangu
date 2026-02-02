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

import torch
import torch_npu

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class OpcheckLinearOperation(operation_test.OperationTest):
    @staticmethod
    def deqscale2int32(scale):
        import numpy as np
        scale = scale.cpu().numpy()
        scale = np.frombuffer(scale.astype(np.int32).tobytes(), dtype=np.float32)
        scale = torch.tensor(scale)
        return scale.npu()

    def golden_calc(self, in_tensors):
        from msit_llm.opcheck.check_case import OutTensorType
        soc_version = self.get_soc_version()
        if soc_version == 'Ascend310P':
            in_tensors[1] = self.convert_data_format(in_tensors[1])

        transpose_a = self.op_param.get("transposeA", False)
        transpose_b = self.op_param.get("transposeB", True)
        has_bias = self.op_param.get("hasBias", False)
        out_data_type = self.op_param.get("outDataType", OutTensorType.ACL_DT_UNDEFINED.value)

        x = in_tensors[0]
        weight = in_tensors[1]
        bias = in_tensors[2] if has_bias else None # 当has_bias = true时才输入
        deq_scale = in_tensors[3] if len(in_tensors) == 4 else None # 反量化的scale，量化场景下才输入

        if transpose_a:
            x = torch.transpose(x, 0, 1) if len(x.shape) == 2 else torch.transpose(x, 1, 2)

        if len(weight.shape) == 4:
            weight = torch.transpose(weight, 1, 2)
            if weight.shape[0] == 1:
                weight = weight.reshape(weight.shape[1], weight.shape[2] * weight.shape[3])
            else:
                weight = weight.reshape(weight.shape[0], weight.shape[1], weight.shape[2] * weight.shape[3])

        if transpose_b:
            weight = torch.transpose(weight, 0, 1) if len(weight.shape) == 2 else torch.transpose(weight, 1, 2)

        if x.dtype == torch.int8:
            x = x.type(torch.float32)
            weight = weight.type(torch.float32)
        golden_result = torch.matmul(x, weight)
        
        if bias is not None:
            golden_result = golden_result + bias
        if deq_scale is not None:
            deq_scale = OpcheckLinearOperation.deqscale2int32(deq_scale)
            golden_result = golden_result * deq_scale

        if out_data_type == OutTensorType.ACL_FLOAT16.value:
            golden_result = golden_result.type(torch.float16)
        elif out_data_type == OutTensorType.ACL_BF16.value:
            golden_result = golden_result.type(torch.bfloat16)

        return [golden_result]

    def test(self):
        ret = self.validate_param("transposeA", "transposeB")
        if not ret:
            return
        self.execute()