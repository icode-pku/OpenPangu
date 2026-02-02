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
from collections import namedtuple
from typing import Sequence
from math import gcd
import numpy as np

NAMEDTUPLE_PRECISION_METRIC = namedtuple('precision_metric', ['abs', 'kl', 'cos_sim'])('abs', 'kl', 'cos_sim')
NAMEDTUPLE_PRECISION_MODE = namedtuple(
    'precision_mode', ["keep_origin_dtype", "force_fp16", "force_fp32"]
)("keep_origin_dtype", "force_fp16", "force_fp32")
    

def broadcast_to_maxshape(shapes: list):
    """
    produce broadcast shape
    for example:
        input: shape is [[2, 3], [3, 2, 1], [3, 1, 3]]
        output: [1, 2, 3], [3, 2, 1], [3, 1, 3], [3, 2, 3]
    """
    def max_dimension_value(_shape):
        no_one_shape = [s for s in _shape if s != 1]
        if len(no_one_shape) == 0:
            max_value = 1
        else:
            max_value = no_one_shape[0]
        return max_value
    max_dim_length = max(len(list(shape)) for shape in shapes)
    input_shapes = []
    for shape in shapes:
        input_shapes.append([1 for _ in range(max_dim_length - len(shape))] + list(shape))
    input_shapes = list(map(list, zip(*input_shapes)))
    max_shape = [max_dimension_value(shape) for shape in input_shapes]
    input_shapes = list(map(list, zip(*input_shapes)))
    return (*input_shapes, max_shape)


def ceil_div(a, b):
    if b == 0:
        raise ValueError("Division by zero is not allowed, Please check!")
    return (a + b - 1) // b


def align(a, b):
    return ceil_div(a, b) * b


def lcm(a, b):
    res_gcd = gcd(a, b)
    if res_gcd == 0:
        raise ValueError("Division by zero is not allowed, Please check!")
    return a * b // res_gcd


def due_fp16_overflow(data):
    """Overflow interception"""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    data = np.nan_to_num(data)
    return data
