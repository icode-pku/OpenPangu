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


class ElewiseType(Enum):
    ELEWISE_UNDEFINED = 0 # 默认值，未定义
    ELEWISE_CAST = 1 # 数据类型转换
    ELEWISE_MULS = 2 # 向量逐元素乘值
    ELEWISE_COS = 3 # 逐元素计算余弦值
    ELEWISE_SIN = 4 # 逐元素计算正弦值
    ELEWISE_NEG = 5 # 逐元素取相反值
    ELEWISE_QUANT = 6 # 量化
    ELEWISE_LOGICAL_NOT = 7 # 逐元素逻辑非
    ELEWISE_ADD = 8 # 逐元素相加
    ELEWISE_MUL = 9 # 向量与向量逐元素相乘
    ELEWISE_REALDIV = 10 # 向量与向量逐元素相除
    ELEWISE_LOGICAL_AND = 11 # 逐元素逻辑与
    ELEWISE_LOGICAL_OR = 12 # 逐元素逻辑或
    ELEWISE_LESS = 13 # 逐元素判断是否小于
    ELEWISE_GREATER = 14 # 逐元素判断是否大于
    ELEWISE_SUB = 15 # 逐元素相减
    ELEWISE_EQUAL = 16 # 逐元素判断是否相等
    ELEWISE_QUANT_PER_CHANNEL = 17 # 每个通道量化
    ELEWISE_DEQUANT_PER_CHANNEL = 18 # 每个通道反量化
    ELEWISE_DYNAMIC_QUANT = 19 # 逐行动态量化
    ELEWISE_TANH = 20 # 逐元素计算双曲正切值


class OpcheckElewiseAddOperation(operation_test.OperationTest):
    def elewise_cast(self, in_tensors):
        from msit_llm.opcheck.check_case import OutTensorType
        out_tensor_type = self.op_param.get('OutTensorType', OutTensorType.ACL_DT_UNDEFINED.value)
        # ELEWISE_CAST输入输出仅支持float/float16/int32/int64数据类型
        out_tensor_type_support_list = [
            OutTensorType.ACL_FLOAT.value,
            OutTensorType.ACL_FLOAT16.value,
            OutTensorType.ACL_INT32.value,
            OutTensorType.ACL_INT64.value
        ]
        self.validate_int_range(out_tensor_type, out_tensor_type_support_list, "OutTensorType")       
        golden_result = in_tensors[0]
        if out_tensor_type == OutTensorType.FLOAT.value:
            golden_result = in_tensors[0].float()
        elif out_tensor_type == OutTensorType.HALF.value:
            golden_result = in_tensors[0].half()
        elif out_tensor_type == OutTensorType.INT.value:
            golden_result = in_tensors[0].int()
        elif out_tensor_type == OutTensorType.LONG.value:
            golden_result = in_tensors[0].long()
        return [golden_result]

    def elewise_muls(self, in_tensors):
        if 'mulsParam' in self.op_param:
            var_attr = self.op_param['mulsParam'].get('varAttr', 0.0)
        else:
            var_attr = self.op_param.get('varAttr', 0.0)
        golden_result = in_tensors[0] * var_attr
        return [golden_result]

    def elewise_cos(self, in_tensors):
        golden_result = torch.cos(in_tensors[0])
        return [golden_result]

    def elewise_sin(self, in_tensors):
        golden_result = torch.sin(in_tensors[0])
        return [golden_result]

    def elewise_neg(self, in_tensors):
        golden_result = in_tensors[0] * (-1.0)
        return [golden_result]

    def elewise_quant(self, in_tensors):
        # ELEWISE_QUANT输出仅支持int8数据类型
        golden_result = in_tensors[0].type(torch.int8)
        return [golden_result]

    def elewise_logical_not(self, in_tensors):
        golden_result = torch.logical_not(in_tensors[0])
        return [golden_result]

    def elewise_add(self, in_tensors):
        golden_result = in_tensors[0] + in_tensors[1]
        return [golden_result]

    def elewise_mul(self, in_tensors):
        golden_result = in_tensors[0] * in_tensors[1]
        return [golden_result]

    def elewise_realdiv(self, in_tensors):
        golden_result = torch.div(in_tensors[0], in_tensors[1])
        return [golden_result]

    def elewise_logical_and(self, in_tensors):
        # ELEWISE_LOGICAL_AND输入输出仅支持int8数据类型
        golden_result = torch.logical_and(in_tensors[0].type(torch.bool), in_tensors[1].type(torch.bool))
        return [golden_result.type(torch.int8)]

    def elewise_logical_or(self, in_tensors):
        # ELEWISE_LOGICAL_OR输入输出仅支持int8数据类型
        golden_result = torch.logical_or(in_tensors[0].type(torch.bool), in_tensors[1].type(torch.bool))
        return [golden_result.type(torch.int8)]

    def elewise_less(self, in_tensors):
        # ELEWISE_LESS输出仅支持int8数据类型
        golden_result = torch.lt(in_tensors[0], in_tensors[1]).type(torch.int8)
        return [golden_result]

    def elewise_greater(self, in_tensors):
        # ELEWISE_GREATER输出仅支持int8数据类型
        golden_result = torch.gt(in_tensors[0], in_tensors[1]).type(torch.int8)
        return [golden_result]

    def elewise_sub(self, in_tensors):
        golden_result = in_tensors[0] - in_tensors[1]
        return [golden_result]

    def elewise_equal(self, in_tensors):
        # ELEWISE_EQUAL输出仅支持int8数据类型
        golden_result = torch.eq(in_tensors[0], in_tensors[1]).type(torch.int8)
        return [golden_result]

    def elewise_quant_per_channel(self, in_tensors):
        # ELEWISE_QUANT_PER_CHANNEL输出仅支持int8数据类型
        input_x = in_tensors[0]
        input_scale = in_tensors[1]
        input_offset = in_tensors[2]

        # input_scale中元素要求不为0。可以为标量
        try:
            scaled_input = torch.round(input_x / input_scale)
        except ZeroDivisionError as e:
            raise RuntimeError("get ZeroDivisionError when calc ELEWISE_QUANT_PER_CHANNEL golden") from e

        if len(input_offset) == 0:
            out = torch.clamp(scaled_input, -128, 127)
        else:
            out = torch.clamp(scaled_input + input_offset, -128, 127)

        return [out.type(torch.int8)]

    def elewise_dequant_per_channel(self, in_tensors):
        # ELEWISE_DEQUANT_PER_CHANNEL输出仅支持float16数据类型
        input_y = in_tensors[0].type(torch.float16) # 支持int8/float16数据类型，计算时需要转换为float16
        input_scale = in_tensors[1] # 仅支持float16数据类型
        input_offset = in_tensors[2].type(torch.float16) # 支持int8/float16数据类型，计算时需要转换为float16
        
        if len(input_offset) == 0:
            out = torch.clamp(input_y * input_scale, -65504, 65504)
        else:
            out = torch.clamp(input_y - input_offset * input_scale, -65504, 65504)

        return [out.type(torch.float16)]

    def elewise_dynamic_quant(self, in_tensors):
        input_x = in_tensors[0]
        shape_input = input_x.shape

        quant_param = self.op_param.get("quantParam", {})
        asymmetric = quant_param.get("asymmetric", False)
        input_scale = quant_param.get("inputScale", 1.0)
        input_offset = quant_param.get("inputOffset", 0)

        if asymmetric:
            row_max = torch.max(input_x, axis=-1, keepdims=True).type(torch.float32)
            row_min = torch.min(input_x, axis=-1, keepdims=True).type(torch.float32)
            out_scale = (row_max - row_min) / 255
            out_offset = - (row_max + row_min) / 2 * out_scale

            input_x = input_x.type(torch.float32)
            input_x = input_x / out_scale + out_offset
            input_x = torch.clamp(input_x, -128, 127)
            out_x = torch.round(input_x)
            return [out_x.type(torch.int8), out_scale.squeeze(axis=-1).type(torch.float32), 
                    out_offset.squeeze(axis=-1).type(torch.float32)]
        else:
            input_abs = torch.abs(input_x)
            scale = torch.max(input_abs, axis=-1, keepdims=True).type(torch.float32)
            out_scale = scale / 127

            input_x = input_x.type(torch.float32)
            input_x = (input_x * 127) / scale
            out_x = torch.round(input_x)
            return [out_x.type(torch.int8), out_scale.squeeze(axis=-1).type(torch.float32)]

    def elewise_tanh(self, in_tensors):
        # ELEWISE_TANH输入输出仅支持float16数据类型 
        golden_result = torch.tanh(in_tensors[0])
        return [golden_result]

    def golden_calc(self, in_tensors):
        elewise_type = self.op_param.get("elewiseType", ElewiseType.ELEWISE_UNDEFINED.value)
        elewise_type_support_list = [
            ElewiseType.ELEWISE_CAST.value,
            ElewiseType.ELEWISE_MULS.value,
            ElewiseType.ELEWISE_COS.value,
            ElewiseType.ELEWISE_SIN.value,
            ElewiseType.ELEWISE_NEG.value,
            ElewiseType.ELEWISE_QUANT.value,
            ElewiseType.ELEWISE_LOGICAL_NOT.value,
            ElewiseType.ELEWISE_ADD.value,
            ElewiseType.ELEWISE_MUL.value,
            ElewiseType.ELEWISE_REALDIV.value,
            ElewiseType.ELEWISE_LOGICAL_AND.value,
            ElewiseType.ELEWISE_LOGICAL_OR.value,
            ElewiseType.ELEWISE_LESS.value,
            ElewiseType.ELEWISE_GREATER.value,
            ElewiseType.ELEWISE_SUB.value,
            ElewiseType.ELEWISE_EQUAL.value,
            ElewiseType.ELEWISE_QUANT_PER_CHANNEL.value,
            ElewiseType.ELEWISE_DEQUANT_PER_CHANNEL.value,
            ElewiseType.ELEWISE_DYNAMIC_QUANT.value,
            ElewiseType.ELEWISE_TANH.value
        ]
        self.validate_int_range(elewise_type, elewise_type_support_list, "elewiseType")

        golden_func = {
            ElewiseType.ELEWISE_CAST.value: self.elewise_cast,
            ElewiseType.ELEWISE_MULS.value: self.elewise_muls,
            ElewiseType.ELEWISE_COS.value: self.elewise_cos,
            ElewiseType.ELEWISE_SIN.value: self.elewise_sin,
            ElewiseType.ELEWISE_NEG.value: self.elewise_neg,
            ElewiseType.ELEWISE_QUANT.value: self.elewise_quant,
            ElewiseType.ELEWISE_LOGICAL_NOT.value: self.elewise_logical_not,
            ElewiseType.ELEWISE_ADD.value: self.elewise_add,
            ElewiseType.ELEWISE_MUL.value: self.elewise_mul,
            ElewiseType.ELEWISE_REALDIV.value: self.elewise_realdiv,
            ElewiseType.ELEWISE_LOGICAL_AND.value: self.elewise_logical_and,
            ElewiseType.ELEWISE_LOGICAL_OR.value: self.elewise_logical_or,
            ElewiseType.ELEWISE_LESS.value: self.elewise_less,
            ElewiseType.ELEWISE_GREATER.value: self.elewise_greater,
            ElewiseType.ELEWISE_SUB.value: self.elewise_sub,
            ElewiseType.ELEWISE_EQUAL.value: self.elewise_equal,
            ElewiseType.ELEWISE_QUANT_PER_CHANNEL.value: self.elewise_quant_per_channel,
            ElewiseType.ELEWISE_DEQUANT_PER_CHANNEL.value: self.elewise_dequant_per_channel,
            ElewiseType.ELEWISE_DYNAMIC_QUANT.value: self.elewise_dynamic_quant,
            ElewiseType.ELEWISE_TANH.value: self.elewise_tanh
        } 
        golden = golden_func.get(elewise_type, self.elewise_cast)(in_tensors)

        return golden

    def test(self):
        ret = self.validate_param("elewiseType")
        if not ret:
            return

        elewise_type = self.op_param.get("elewiseType", None)
        msg = f"elewiseType: {elewise_type}"
        logger.debug(msg)

        self.execute()