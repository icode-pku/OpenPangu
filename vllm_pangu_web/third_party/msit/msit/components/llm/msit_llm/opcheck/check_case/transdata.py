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


class TransdataType(Enum):
    UNDEFINED = 0 # 默认
    FRACTAL_NZ_TO_ND = 1 # FRACTAL_NZ转ND
    ND_TO_FRACTAL_NZ = 2 # ND转FRACTAL_NZ 


class OpcheckTransdataOperation(operation_test.OperationTest):
    @staticmethod
    def round_up(x, align):
        if align == 0:
            return -1
        return (x + align - 1) // align * align

    @staticmethod
    def custom_pad(x, pad_dims):
        return torch.nn.functional.pad(x, pad_dims)

    @staticmethod
    def custom_reshape(x, target_shape):
        return x.reshape(target_shape)

    @staticmethod
    def custom_transpose(x, dim1, dim2):
        return x.transpose(dim1, dim2)

    @staticmethod
    def golden_nd_to_nz_3d(in_tensors):
        align_int8 = 32
        default_align = 16

        size0 = in_tensors[0].size(0)
        size1 = in_tensors[0].size(1)
        size2 = in_tensors[0].size(2)

        aux_dims = [0, 0, 0, 0]
        aux_dims[0] = size0
        aux_dims[1] = OpcheckTransdataOperation.round_up(size1, default_align)

        pad_dims = [0, 0, 0, 0]  
        pad_dims[3] = OpcheckTransdataOperation.round_up(size1, default_align) - size1

        if in_tensors[0].dtype == torch.int8:
            aux_dims[2] = OpcheckTransdataOperation.round_up(size2, align_int8) // align_int8
            aux_dims[3] = align_int8
            pad_dims[1] = OpcheckTransdataOperation.round_up(size2, align_int8) - size2
        else:
            aux_dims[2] = OpcheckTransdataOperation.round_up(size2, default_align) // default_align
            aux_dims[3] = default_align
            pad_dims[1] = OpcheckTransdataOperation.round_up(size2, default_align) - size2

        return OpcheckTransdataOperation.custom_transpose(
                    OpcheckTransdataOperation.custom_reshape(
                        OpcheckTransdataOperation.custom_pad(in_tensors[0], pad_dims),
                        aux_dims
                    ),
                    1, 2
                ).contiguous()

    @staticmethod
    def golden_nd_to_nz_2d(in_tensors):
        align_int8 = 32
        default_align = 16

        size0 = in_tensors[0].size(0)
        size1 = in_tensors[0].size(1)

        aux_dims = [0, 0, 0, 0]
        aux_dims[0] = 1
        aux_dims[1] = OpcheckTransdataOperation.round_up(size0, default_align)

        pad_dims = [0, 0, 0, 0]  
        pad_dims[3] = OpcheckTransdataOperation.round_up(size0, default_align) - size0

        if in_tensors[0].dtype == torch.int8:
            aux_dims[2] = OpcheckTransdataOperation.round_up(size1, align_int8) // align_int8
            aux_dims[3] = align_int8
            pad_dims[1] = OpcheckTransdataOperation.round_up(size1, align_int8) - size1
        else:
            aux_dims[2] = OpcheckTransdataOperation.round_up(size1, default_align) // default_align
            aux_dims[3] = default_align
            pad_dims[1] = OpcheckTransdataOperation.round_up(size1, default_align) - size1

        return OpcheckTransdataOperation.custom_transpose(
                    OpcheckTransdataOperation.custom_reshape(
                        OpcheckTransdataOperation.custom_pad(in_tensors[0], pad_dims),
                        aux_dims
                    ),
                    1, 2
                ).contiguous()

    @staticmethod
    def golden_nz_to_nd(in_tensors, out_crops):
        size0 = in_tensors[0].size(0)
        size1 = in_tensors[0].size(1)
        size2 = in_tensors[0].size(2)
        size3 = in_tensors[0].size(3)

        aux_dims = [0, 0, 0]
        aux_dims[0] = size0
        aux_dims[1] = size2
        aux_dims[2] = size1 * size3

        return OpcheckTransdataOperation.custom_reshape(
                    OpcheckTransdataOperation.custom_transpose(in_tensors[0], 1, 2),
                    aux_dims
                )[:, :out_crops[0], :out_crops[1]]

    def golden_calc(self, in_tensors):
        transdata_type = self.op_param.get("transdataType", TransdataType.UNDEFINED.value)
        if transdata_type == TransdataType.ND_TO_FRACTAL_NZ.value:
            if len(in_tensors[0].size()) == 3:
                golden_result = OpcheckTransdataOperation.golden_nd_to_nz_3d(in_tensors)
            else:
                golden_result = OpcheckTransdataOperation.golden_nd_to_nz_2d(in_tensors)
        else:
            out_crops = self.op_param.get("outCrops", None)
            golden_result = OpcheckTransdataOperation.golden_nz_to_nd(in_tensors, out_crops)

        return [golden_result]

    def test(self):
        ret = self.validate_param("transdataType")
        if not ret:
            return
        self.execute()