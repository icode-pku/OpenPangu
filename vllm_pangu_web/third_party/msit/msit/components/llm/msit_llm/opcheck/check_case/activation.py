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

from enum import Enum
import torch
import torch_npu
import torch.nn.functional as F

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class ActivationType(Enum):
    ACTIVATION_UNDEFINED = 0 # 未定义
    ACTIVATION_RELU = 1
    ACTIVATION_GELU = 2
    ACTIVATION_FAST_GELU = 3
    ACTIVATION_SWISH = 4
    ACTIVATION_LOG = 5
    ACTIVATION_SWIGLU_FORWARD = 6
    ACTIVATION_SWIGLU_BACKWARD = 7
    ACTIVATION_MAX = 8 # 枚举最大值（暂不支持）


class GeLUMode(Enum):
    TANH_MODE = 0 # 默认值，使用tanh估算
    NONE_MODE = 1 # 原GeLU计算公式


class ActivationGolden:
    @staticmethod
    def relu_golden(in_tensors, _):
        return torch.nn.functional.relu(in_tensors)

    @staticmethod
    def gelu_golden(in_tensors, gelu_mode):
        approx = "tanh" if gelu_mode == GeLUMode.TANH_MODE.value else "none"
        return torch.nn.functional.gelu(in_tensors, approximate=approx)

    @staticmethod
    def fast_gelu_golden(in_tensors, _):
        in_tensors = in_tensors.float()
        try:
            float_result = in_tensors * torch.exp(0.851 * (in_tensors - torch.abs(in_tensors))) / (1 + 
                            torch.exp(-1.702 * torch.abs(in_tensors)))
        except ZeroDivisionError as e:
            raise RuntimeError("get ZeroDivisionError when calc ACTIVATION_FAST_GELU golden") from e
        return float_result.half()

    @staticmethod
    def swish_golden(in_tensors, scale):
        in_tensors = in_tensors.float()
        try:
            float_result = in_tensors / (1 + torch.exp(-in_tensors * scale))
        except ZeroDivisionError as e:
            raise RuntimeError("get ZeroDivisionError when calc ACTIVATION_SWISH golden") from e
        return float_result.half()

    @staticmethod
    def log_golden(in_tensors, _):
        return torch.log(in_tensors)

    @staticmethod
    def swigluforward_golden(in_tensors, dim):
        dtype = in_tensors.dtype
        float_in_tensors = in_tensors.float()
        a, b = float_in_tensors.chunk(2, dim)
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        float_result = F.silu(a) * b
        return float_result.to(dtype)

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    @staticmethod
    def swish_grad(x):
        return torch.sigmoid(x) + x * (1 - torch.sigmoid(x)) * torch.sigmoid(x)

    @staticmethod
    def swiglubackward_golden(in_tensors, dim):
        dtype = in_tensors[1].dtype
        tensor_y_grad = in_tensors[1].float()
        x = in_tensors[1].float()
        a, b = x.chunk(2, dim)
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        y1 = b * tensor_y_grad * ActivationGolden.swish_grad(a)
        y2 = tensor_y_grad * ActivationGolden.swish(a)
        y = torch.concat((y1, y2), dim)
        return y.to(dtype)        


class OpcheckActivationOperation(operation_test.OperationTest):
    golden_func = {
        1: ActivationGolden.relu_golden,
        2: ActivationGolden.gelu_golden,
        3: ActivationGolden.fast_gelu_golden,
        4: ActivationGolden.swish_golden,
        5: ActivationGolden.log_golden,
        6: ActivationGolden.swigluforward_golden,
        7: ActivationGolden.swiglubackward_golden,
    } 

    def golden_calc(self, in_tensors):
        activation_type = self.op_param.get("activationType", ActivationType.ACTIVATION_UNDEFINED.value)
        activation_type_support_list = [
            ActivationType.ACTIVATION_RELU.value,
            ActivationType.ACTIVATION_GELU.value,
            ActivationType.ACTIVATION_FAST_GELU.value,
            ActivationType.ACTIVATION_SWISH.value,
            ActivationType.ACTIVATION_LOG.value,
            ActivationType.ACTIVATION_SWIGLU_FORWARD.value,
            ActivationType.ACTIVATION_SWIGLU_BACKWARD.value
        ]
        self.validate_int_range(activation_type, activation_type_support_list, "activationType")
        scale = self.op_param.get("scale", 1.0)
        dim = self.op_param.get("dim", -1)
        if activation_type == ActivationType.ACTIVATION_SWIGLU_FORWARD.value:
            golden_result = OpcheckActivationOperation.golden_func[activation_type](in_tensors[0], dim)
        elif activation_type == ActivationType.ACTIVATION_SWIGLU_BACKWARD.value:
            golden_result = OpcheckActivationOperation.golden_func[activation_type](in_tensors, dim)
        elif activation_type == ActivationType.ACTIVATION_GELU.value:
            gelu_mode = self.op_param.get("geluMode", GeLUMode.TANH_MODE.value)
            self.validate_int_range(gelu_mode, [GeLUMode.TANH_MODE.value, GeLUMode.NONE_MODE.value], "geluMode")
            golden_result = OpcheckActivationOperation.golden_func[activation_type](in_tensors[0], gelu_mode)
        else:
            golden_result = OpcheckActivationOperation.golden_func[activation_type](in_tensors[0], scale)
        return [golden_result]

    def test(self):
        ret = self.validate_param("activationType")
        if not ret:
            return
        self.execute()