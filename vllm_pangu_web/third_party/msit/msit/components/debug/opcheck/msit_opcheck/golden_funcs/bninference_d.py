#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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

import numpy as np

from msit_opcheck.utils import broadcast_to_maxshape
from msit_opcheck.operation_test import OperationTest
from msit_opcheck.constants import FLOAT32, FLOAT16, BFLOAT16

SCALE_INDEX = 3
BIAS_INDEX = 4


def _fused_scale_bias_compute(x, mean, variance, scale, bias):
    shape_x = x.shape
    shape_mean = mean.shape
    shape_list = broadcast_to_maxshape([shape_x, shape_mean])
    x_broadcast = np.broadcast_to(x, shape_list[-1])
    mean_broadcast = np.broadcast_to(mean, shape_list[-1])
    var_broadcast = np.broadcast_to(variance, shape_list[-1])
    mean_add = np.add(x_broadcast, mean_broadcast)
    res_y = np.multiply(var_broadcast, mean_add)
    scale_broad = np.broadcast_to(scale, shape_list[-1])
    bias_broad = np.broadcast_to(bias, shape_list[-1])
    res_y = res_y.astype(FLOAT32)
    scale_broad = scale_broad.astype(FLOAT32)
    res_tmp = np.multiply(res_y, scale_broad)
    return np.add(res_tmp, bias_broad)


def _fused_scale_compute(x, mean, variance, scale):
    shape_x = x.shape
    shape_mean = mean.shape
    shape_list = broadcast_to_maxshape([shape_x, shape_mean])
    x_broadcast = np.broadcast_to(x, shape_list[-1])
    mean_broadcast = np.broadcast_to(mean, shape_list[-1])
    var_broadcast = np.broadcast_to(variance, shape_list[-1])
    mean_add = np.add(x_broadcast, mean_broadcast)
    res_y = np.multiply(var_broadcast, mean_add)
    scale_broad = np.broadcast_to(scale, shape_list[-1])
    res_y = res_y.astype(FLOAT32, copy=False)
    scale_broad = scale_broad.astype(FLOAT32, copy=False)
    return np.multiply(res_y, scale_broad)


def _fused_compute(x, mean, variance):
    shape_x = x.shape
    shape_mean = mean.shape
    shape_list = broadcast_to_maxshape([shape_x, shape_mean])
    x_broadcast = np.broadcast_to(x, shape_list[-1])
    mean_broadcast = np.broadcast_to(mean, shape_list[-1])
    var_broadcast = np.broadcast_to(variance, shape_list[-1])
    mean_add = np.add(x_broadcast, mean_broadcast)
    return np.multiply(var_broadcast, mean_add)


class BnInferenceOperation(OperationTest):
    def golden_calc(self, in_tensors):
        x, mean, variance = in_tensors[:3]
        scale = in_tensors[SCALE_INDEX] if len(in_tensors) > SCALE_INDEX else None
        bias = in_tensors[BIAS_INDEX] if len(in_tensors) > BIAS_INDEX else None
        input_format = self.op_param['input_desc'][0]['layout']
        dtype = x.dtype
        if str(dtype) == BFLOAT16:  # tf.bfloat16.as_numpy_dtype:
            x = x.astype(FLOAT32)
            mean = mean.astype(FLOAT32)
            variance = variance.astype(FLOAT32)
            if scale is not None:
                scale = scale.astype(FLOAT32)
            if bias is not None:
                bias = bias.astype(FLOAT32)
        variance.shape = mean.shape
        if scale is not None and bias is not None:
            scale.shape = mean.shape
            bias.shape = mean.shape
            res = _fused_scale_bias_compute(x, mean, variance, scale, bias)
        elif scale is not None and bias is None:
            scale.shape = mean.shape
            res = _fused_scale_compute(x, mean, variance, scale)
        else:
            res = _fused_compute(x, mean, variance)
        return [res.astype(dtype, copy=False)]

    def test_bninference(self):
        self.execute()
