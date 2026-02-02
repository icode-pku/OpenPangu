# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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


from msit_opcheck.golden_funcs.logical_or import LogicalOrOperation
from msit_opcheck.golden_funcs.logical_and import LogicalAndOperation
from msit_opcheck.golden_funcs.pad import PadOperation
from msit_opcheck.golden_funcs.add_n import AddOperation
from msit_opcheck.golden_funcs.batchnorm import BatchNormOperation
from msit_opcheck.golden_funcs.cast import CastOperation
from msit_opcheck.golden_funcs.bias_add import BiasAddOperation
from msit_opcheck.golden_funcs.softmax import SoftmaxOperation 
from msit_opcheck.golden_funcs.mat_mul import MatmulOperation
from msit_opcheck.golden_funcs.bninference_d import BnInferenceOperation
from msit_opcheck.golden_funcs.concat_d import ConcatOperation
from msit_opcheck.golden_funcs.conv2d import Conv2dOperation
from msit_opcheck.golden_funcs.gather_v2 import GatherOperation
from msit_opcheck.golden_funcs.minimum import MinimumOperation
from msit_opcheck.golden_funcs.pack import PackOperation
from msit_opcheck.golden_funcs.reduce import ReduceSumOperation, ReduceMeanOperation
from msit_opcheck.golden_funcs.relu import ReluOperation
from msit_opcheck.golden_funcs.sigmoid import SigmoidOperation
from msit_opcheck.golden_funcs.tanh import TanhOperation
from msit_opcheck.golden_funcs.tile_d import TileDOperation
from msit_opcheck.golden_funcs.transpose import TransposeOperation
from msit_opcheck.golden_funcs.mul import MulOperation
from msit_opcheck.golden_funcs.select import SelectOperation
from msit_opcheck.golden_funcs.clip_by_value import ClipByValueOperation
from msit_opcheck.golden_funcs.rsqrt import RsqrtOperation
from msit_opcheck.golden_funcs.less import LessOperation
from msit_opcheck.golden_funcs.sub import SubOperation
from msit_opcheck.golden_funcs.stridedslice import StridedSliceOperation
from msit_opcheck.golden_funcs.batch_matmul import BatchMatMulOperation

OP_DICT = dict({"Pad": PadOperation, 
                "PadD": PadOperation, 
                "LogicalOr": LogicalOrOperation, 
                "Adds": AddOperation,
                "Add": AddOperation, 
                "BatchNorm": BatchNormOperation, 
                "Cast": CastOperation, 
                "SoftmaxV2": SoftmaxOperation, 
                "BiasAdd": BiasAddOperation, 
                "MatMulV2": MatmulOperation,
                "BNInfer": BnInferenceOperation,
                "ConcatV2": ConcatOperation,
                "Conv2D": Conv2dOperation,
                "GatherV2": GatherOperation,
                "LogicalAnd": LogicalAndOperation,
                "Minimum": MinimumOperation,
                "Pack": PackOperation,
                "ReduceSum": ReduceSumOperation,
                "ReduceMean": ReduceMeanOperation,
                "Relu": ReluOperation,
                "Sigmoid": SigmoidOperation,
                "Tanh": TanhOperation,
                "Tile": TileDOperation,
                "Transpose": TransposeOperation,
                "Mul": MulOperation,
                "Select": SelectOperation,
                "ClipByValue": ClipByValueOperation,
                "Rsqrt": RsqrtOperation,
                "Less": LessOperation,
                "Sub": SubOperation,
                "StridedSlice": StridedSliceOperation,
                "BatchMatMulV2": BatchMatMulOperation,
                })
