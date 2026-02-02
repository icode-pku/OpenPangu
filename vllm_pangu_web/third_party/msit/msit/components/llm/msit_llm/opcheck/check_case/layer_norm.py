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

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class LayerNormType(Enum):
    LAYER_NORM_UNDEFINED = 0 # 默认值，未定义
    LAYER_NORM_NROM = 1 # norm
    LAYER_NORM_PRENORM = 2 # prenorm
    LAYER_NORM_POSTNORM = 3 # postnorm
    LAYER_NORM_MAX = 4 # 枚举最大值，暂不支持


class QuantType(Enum):
    QUANT_TYPE_UNDEFINED = -1 # 默认值
    QUANT_TYPE_PER_TENSOR = 0 # 对整个张量进行量化
    QUANT_TYPE_PER_CHANNEL = 1 # 对张量中每个channel分别进行量化
    QUANT_TYPE_PER_GROUP = 2 # 将张量按quantGroupSize划分后，分别进行量化
    QUANT_TYPE_MAX = 3 # 枚举类型最大值，暂不支持


class DynamicQuantType(Enum):
    DYNAMIC_QUANT_UNDEFINED = 0 # 非动态量化
    DYNAMIC_QUANT_SYMMETRIC = 1 # 对称动态量化
    DYNAMIC_QUANT_ASYMMETRIC = 2 # 非对称动态量化，暂不支持


class OpcheckLayerNormOperation(operation_test.OperationTest):
    def layer_norm_quant(self, layer_norm_res, quant_scale, quant_offset):
        golden_result_quant = layer_norm_res * quant_scale + quant_offset
        golden_result_quant = torch.round(golden_result_quant)
        golden_result_quant = torch.clamp(golden_result_quant, -128, 127)
        return golden_result_quant.type(torch.int8)

    def golden_func_norm(self, in_tensors, cur_param):
        eps = cur_param.get('epsilon', 1e-5)
        quant_type = cur_param.get('quantType', QuantType.QUANT_TYPE_UNDEFINED.value)
        is_quant = quant_type != QuantType.QUANT_TYPE_UNDEFINED.value
        dynamic_quant_type = cur_param.get('dynamicQuantType', DynamicQuantType.DYNAMIC_QUANT_UNDEFINED.value)

        layer_input, weight, bias = in_tensors[:3]

        quant_scale = 1
        quant_offset = 0
        if is_quant and dynamic_quant_type == DynamicQuantType.DYNAMIC_QUANT_UNDEFINED.value:
            quant_scale = in_tensors[3]
            quant_offset = in_tensors[4]

        if not is_quant:
            begin_norm_axis = cur_param.get('beginNormAxis', 0) # 归一化的维度，从第几维开始norm
            begin_params_axis = cur_param.get('beginParamsAxis', 0) # 归一化的维度，决定从第几维开始把后面的维度按轴合并
            normalized_shape = layer_input.shape[begin_norm_axis:]
            golden_result = torch.nn.functional.layer_norm(layer_input, normalized_shape, weight, bias, eps)
            return [golden_result]

        if dynamic_quant_type != DynamicQuantType.DYNAMIC_QUANT_UNDEFINED.value:
            layer_norm_result = torch.nn.functional.layer_norm(layer_input, weight.shape, weight, bias, eps)
            dynamic_quant_x = layer_norm_result.cpu()
            if dynamic_quant_type == DynamicQuantType.DYNAMIC_QUANT_SYMMETRIC.value:
                input_abs = torch.abs(dynamic_quant_x)
                scale = torch.max(input_abs, axis=-1, keepdims=True).type(torch.float32)
                dynamic_quant_scale = scale / 127
                dynamic_quant_x = dynamic_quant_x * 127 / scale
                dynamic_quant_y = torch.round(dynamic_quant_x)
                return [dynamic_quant_y.type(torch.int8), 
                        dynamic_quant_scale.squeeze(axis=-1).type(torch.float32)]
            if dynamic_quant_type == DynamicQuantType.DYNAMIC_QUANT_ASYMMETRIC.value:
                row_max = torch.max(dynamic_quant_x, axis=-1, keepdims=True).type(torch.float32)
                row_min = torch.min(dynamic_quant_x, axis=-1, keepdims=True).type(torch.float32)
                dynamic_quant_scale = (row_max - row_min) / 255
                if dynamic_quant_scale == 0:
                    raise ZeroDivisionError("LayerNorm Result is abnormal. Please check!")
                dynamic_quant_offset = - (row_max + row_min) / (2 * dynamic_quant_scale)
                if dynamic_quant_offset == 0:
                    raise ZeroDivisionError("LayerNorm Result is abnormal. Please check!")
                dynamic_quant_x = dynamic_quant_x.type(torch.float32)
                dynamic_quant_x = dynamic_quant_x / dynamic_quant_scale
                dynamic_quant_x = dynamic_quant_x + dynamic_quant_offset
                dynamic_quant_x = torch.clip(dynamic_quant_x, -128, 127)
                dynamic_quant_y = torch.round(dynamic_quant_x)
                return [dynamic_quant_y.type(torch.int8), 
                        dynamic_quant_scale.squeeze(axis=-1).type(torch.float32), 
                        dynamic_quant_offset.squeeze(axis=-1).type(torch.float32)]
        normalized_shape = (1, layer_input.shape[-1])
        layer_norm_res = torch.nn.functional.layer_norm(layer_input, normalized_shape, weight, bias, eps)
        golden_result_quant = self.layer_norm_quant(layer_norm_res, quant_offset, quant_offset)
        return [golden_result_quant]

    def golden_func_prenorm(self, in_tensors, cur_param):
        eps = cur_param.get('epsilon', 1e-5)
        weight = in_tensors[2]
        bias = in_tensors[3]
        normalized_shape = (1, in_tensors[0].shape[-1])
        zoom_scale_value = cur_param.get('zoomScaleValue', 1.0)
        layer_input = torch.add(in_tensors[0], zoom_scale_value * in_tensors[1])
        golden_result = torch.nn.functional.layer_norm(layer_input, normalized_shape, weight, bias, eps)
        return [golden_result, layer_input]

    def golden_func_postnorm(self, in_tensors, cur_param):
        eps = cur_param.get('epsilon', 1e-5)
        quant_type = cur_param.get('quantType', QuantType.QUANT_TYPE_UNDEFINED.value)
        is_quant = quant_type != QuantType.QUANT_TYPE_UNDEFINED.value
        dynamic_quant_type = cur_param.get('dynamicQuantType', DynamicQuantType.DYNAMIC_QUANT_UNDEFINED.value)

        quant_scale = 1
        quant_offset = 0
        quant_alpha = 1
        if is_quant and dynamic_quant_type == DynamicQuantType.DYNAMIC_QUANT_UNDEFINED.value:
            quant_scale = in_tensors[4]
            quant_offset = in_tensors[5]

        weight = in_tensors[2]
        bias = in_tensors[3]
        normalized_shape = (1, in_tensors[0].shape[-1])
        if not is_quant:
            zoom_scale_value = cur_param.get('zoomScale', 1.0)
            layer_input = torch.add(in_tensors[0], zoom_scale_value * in_tensors[1])
            golden_result = torch.nn.functional.layer_norm(layer_input, normalized_shape, weight, bias, eps)
            return [golden_result]
        else:
            layer_input = torch.add(in_tensors[0], in_tensors[1])
            if len(weight.shape) == 1:
                weight = weight.unsqueeze(0)
            if len(bias.shape) == 1:
                bias = bias.unsqueeze(0)
            layer_norm_res = torch.nn.functional.layer_norm(layer_input, normalized_shape, weight, bias, eps)
            golden_result = layer_norm_res * quant_alpha
            golden_result_quant = self.layer_norm_quant(layer_norm_res, quant_offset, quant_offset)
            return [golden_result, golden_result_quant]

    def golden_calc(self, in_tensors):
        layer_type = self.op_param.get('layerType', LayerNormType.LAYER_NORM_UNDEFINED.value)
        layer_type_support_list = [
            LayerNormType.LAYER_NORM_NROM.value,
            LayerNormType.LAYER_NORM_PRENORM.value,
            LayerNormType.LAYER_NORM_POSTNORM.value,
        ]
        self.validate_int_range(layer_type, layer_type_support_list, "layerType")        
        if layer_type == LayerNormType.LAYER_NORM_NROM.value:
            cur_param = self.op_param.get('normParam', None)
        elif layer_type == LayerNormType.LAYER_NORM_PRENORM.value:
            cur_param = self.op_param.get('preNormParam', None)
        elif layer_type == LayerNormType.LAYER_NORM_POSTNORM.value:
            cur_param = self.op_param.get('postNormParam', None)

        if layer_type == LayerNormType.LAYER_NORM_NROM.value:
            golden = self.golden_func_norm(in_tensors, cur_param)
        elif layer_type == LayerNormType.LAYER_NORM_PRENORM.value:
            golden = self.golden_func_prenorm(in_tensors, cur_param)
        elif layer_type == LayerNormType.LAYER_NORM_POSTNORM.value:
            golden = self.golden_func_postnorm(in_tensors, cur_param)

        return golden

    def test(self):
        ret = self.validate_param("layerType")
        if not ret:
            return
        self.execute()