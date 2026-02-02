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

from msit_llm.opcheck import operation_test


class RmsNormType(Enum):
    RMS_NORM_UNDEFINED = 0 # 默认值，未定义
    RMS_NORM_NORM = 1 # NORM参数
    RMS_NORM_PRE_NORM = 2 # PRENORM参数
    RMS_NORM_POST_NORM = 3 # POSTNORM参数


class ModelType(Enum):
    LLAMA_MODEL = 0 # 默认值，使用Llama rmsnorm的公式
    GEMMA_MODEL = 1 # 使用Gemma rmsnorm的公式


class PrecisionMode(Enum):
    HIGH_PRECISION_MODE = 0 # 中间计算使用fp32类型
    HIGH_PERFORMANCE_MODE = 1 # 中间计算使用fp16类型


class QuantType(Enum):
    QUANT_UNDEFINED = 0 # 不量化
    QUANT_INT4 = 1 # 当前不支持
    QUANT_INT8 = 2 # in8量化
    QUANT_INT16 = 3 # 当前不支持
    QUANT_FLOAT8 = 4 # 当前不支持
    QUANT_FLOAT16 = 5 # 当前不支持


class DynamicQuantType(Enum):
    DYNAMIC_QUANT_UNDEFINED = 0 # 非动态量化
    DYNAMIC_QUANT_SYMMETRIC = 1 # 对称动态量化
    DYNAMIC_QUANT_ASYMMETRIC = 2 # 非对称动态量化，暂不支持


class OpcheckRmsNormOperation(operation_test.OperationTest):
    @staticmethod
    def rms_norm_quant_with_tensor(golden_output, beta, scale, offset):
        golden_output = golden_output.float()
        beta = beta.float()
        scale = scale.half()
        golden_output = golden_output + beta
        try:
            golden_output = golden_output / scale + offset
        except ZeroDivisionError as e:
            raise RuntimeError("get ZeroDivisionError when calc RmsNormOperation golden") from e
        golden_output = torch.clamp(golden_output, -128, 127)
        golden_result_quant = torch.round(golden_output)
        return golden_result_quant.type(torch.int8)

    def rms_norm_quant(self, golden_output, in_tensors, cur_param):
        dynamic_quant_type = cur_param.get('dynamicQuantType', DynamicQuantType.DYNAMIC_QUANT_UNDEFINED.value)
        if dynamic_quant_type == DynamicQuantType.DYNAMIC_QUANT_UNDEFINED.value:
            beta, scale, offset = in_tensors[2], in_tensors[3], in_tensors[4]
            golden_result = [OpcheckRmsNormOperation.rms_norm_quant_with_tensor(golden_output, beta, scale, offset)]
        else:
            golden_output = golden_output + in_tensors[2]
            dynamic_quant_x = golden_output.cpu()
            if dynamic_quant_type == DynamicQuantType.DYNAMIC_QUANT_SYMMETRIC.value:
                input_abs = torch.abs(dynamic_quant_x)
                scale = torch.max(input_abs, axis=-1, keepdim=True)
                dynamic_quant_scale = scale / 127
                dynamic_quant_x = dynamic_quant_x * 127 / scale
                dynamic_quant_y = torch.round(dynamic_quant_x)
                golden_result = [dynamic_quant_y.type(torch.int8), dynamic_quant_scale.squeeze(-1).type(torch.float32)]
            elif dynamic_quant_type == DynamicQuantType.DYNAMIC_QUANT_ASYMMETRIC.value:
                row_max = torch.max(dynamic_quant_x, axis=-1, keepdim=True)
                row_min = torch.min(dynamic_quant_x, axis=-1, keepdim=True)
                dynamic_quant_scale = (row_max - row_min) / 255
                dynamic_quant_offset = - (row_max + row_min) / (2 * dynamic_quant_scale)

                dynamic_quant_x = dynamic_quant_x / dynamic_quant_scale + dynamic_quant_offset
                dynamic_quant_x = torch.clamp(dynamic_quant_x, -128, 127)
                dynamic_quant_y = torch.round(dynamic_quant_x)
                golden_result = [
                    dynamic_quant_y.type(torch.int8), 
                    dynamic_quant_scale.squeeze(-1).type(torch.float32), 
                    dynamic_quant_offset.squeeze(-1).type(torch.float32)
                ]
        return golden_result

    def get_golden_result(self, in_tensors, cur_param, layer_type, golden_output, x):
        quant_type = cur_param.get('quantType', QuantType.QUANT_UNDEFINED.value)
        if layer_type == RmsNormType.RMS_NORM_PRE_NORM.value and quant_type == QuantType.QUANT_INT8.value:
            beta, scale, offset = in_tensors[3], in_tensors[4], in_tensors[5]
            golden_result = [OpcheckRmsNormOperation.rms_norm_quant_with_tensor(golden_output, beta, scale, offset), x]
        elif layer_type == RmsNormType.RMS_NORM_NORM.value and quant_type == QuantType.QUANT_INT8.value:
            golden_result = self.rms_norm_quant(golden_output, in_tensors, cur_param)
        elif layer_type == RmsNormType.RMS_NORM_PRE_NORM.value and quant_type == QuantType.QUANT_UNDEFINED.value:
            golden_result = [golden_result.half(), x.half()]
        else:
            golden_result = [golden_output.half()]
        return golden_result

    def validate_rmsnorm_param(self, layer_type):
        layer_type_support_list = [
            RmsNormType.RMS_NORM_NORM.value,
            RmsNormType.RMS_NORM_PRE_NORM.value,
            RmsNormType.RMS_NORM_POST_NORM.value,
        ]
        self.validate_int_range(layer_type, layer_type_support_list, "layerType")

        if layer_type == RmsNormType.RMS_NORM_NORM.value:
            cur_param = self.op_param.get('normParam', None)
        elif layer_type == RmsNormType.RMS_NORM_PRE_NORM.value:
            cur_param = self.op_param.get('preNormParam', None)
        elif layer_type == RmsNormType.RMS_NORM_POST_NORM.value:
            cur_param = self.op_param.get('postNormParam', None)

        quant_type = cur_param.get('quantType', QuantType.QUANT_UNDEFINED.value)
        quant_type_support_list = [QuantType.QUANT_UNDEFINED.value, QuantType.QUANT_INT8.value]
        self.validate_int_range(quant_type, quant_type_support_list, "quantType")
        return cur_param

    def golden_calc(self, in_tensors):
        layer_type = self.op_param.get('layerType', RmsNormType.RMS_NORM_UNDEFINED.value)
        cur_param = self.validate_rmsnorm_param(layer_type)
        quant_type = cur_param.get('quantType', QuantType.QUANT_UNDEFINED.value)

        eps = cur_param.get('epsilon', 1e-5)
        layer_norm_eps = cur_param.get('layerNormEpsilon', 1e-5) # 暂时不使用
        x, gamma = in_tensors[0].float(), in_tensors[1].float().view(1, -1)
        if layer_type == RmsNormType.RMS_NORM_PRE_NORM.value and quant_type == QuantType.QUANT_INT8.value:
            x = x + in_tensors[1].float()
            gamma = in_tensors[2].float()
        if layer_type == RmsNormType.RMS_NORM_POST_NORM.value or \
            (layer_type == RmsNormType.RMS_NORM_PRE_NORM.value and quant_type == QuantType.QUANT_UNDEFINED.value):
            x = x + in_tensors[1].float()
            has_bias = cur_param.get('hasBias', False)
            if has_bias:
                x = x + in_tensors[2].float()
            gamma = in_tensors[3].float() if has_bias else in_tensors[2].float()
        model_type = cur_param.get('modelType', ModelType.LLAMA_MODEL.value)
        gamma = 1 + gamma if model_type == ModelType.GEMMA_MODEL.value else gamma
        try:
            norm = torch.sum(x / float(gamma.size(-1)) * x, dim=-1, keepdim=True) + eps
        except ZeroDivisionError as e:
            raise RuntimeError("get ZeroDivisionError when calc RmsNormOperation golden") from e

        precision_mode = cur_param.get('precisionMode', PrecisionMode.HIGH_PRECISION_MODE.value)
        is_rstd = cur_param.get("rstd", False)
        if layer_type == RmsNormType.RMS_NORM_NORM.value and is_rstd:
            gamma, reduce_dims, edim = in_tensors[1].float(), [], x.dim() - gamma.dim()
            for i in range(gamma.dim()):
                reduce_dims.append(edim + i)
            rstd = torch.sqrt(x.pow(2).mean(reduce_dims, keepdim=True) + eps)
            return [(x * rstd) * gamma, rstd]
        try:
            if precision_mode == PrecisionMode.HIGH_PERFORMANCE_MODE.value:
                golden_output = (x / torch.sqrt(norm)).half() * gamma.to(torch.float16).view(1, -1)
            else:
                golden_output = x * gamma / torch.sqrt(norm)
        except ZeroDivisionError as e:
            raise RuntimeError("get ZeroDivisionError when calc RmsNormOperation golden") from e

        golden_result = self.get_golden_result(in_tensors, cur_param, layer_type, golden_output, x)
        return golden_result

    def test(self):
        ret = self.validate_param("layerType")
        if not ret:
            return
        self.execute()