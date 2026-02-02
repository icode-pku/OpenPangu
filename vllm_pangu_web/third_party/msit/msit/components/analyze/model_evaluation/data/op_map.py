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

import os
from typing import Dict

import yaml

from model_evaluation.common import utils
from model_evaluation.common.enum import Framework
from components.utils.file_open_check import ms_open
from components.utils.constants import TENSOR_MAX_SIZE


class OpMap:
    def __init__(self, op_map: Dict[str, str]) -> None:
        self._op_map: Dict[str, str] = op_map

    @classmethod
    def load_op_map(cls, framework) -> 'OpMap':
        cur_dir: str = os.path.dirname(os.path.realpath(__file__))
        sub_path: str = os.path.join(cur_dir, 'map_conf')
        if not os.path.isdir(sub_path):
            raise RuntimeError(f'{sub_path} is not dir.')

        framework_yaml_map = {Framework.CAFFE: 'caffe.yaml', Framework.ONNX: 'onnx.yaml', Framework.TF: 'tf.yaml'}
        if framework not in framework_yaml_map:
            raise RuntimeError('framework is invalid.')
        yaml_path = os.path.join(sub_path, framework_yaml_map.get(framework))
        if not os.path.exists(yaml_path):
            raise RuntimeError(f'{yaml_path} not exist.')
        if not utils.check_file_security(yaml_path):
            raise RuntimeError(f'check file security error, file:{yaml_path}')

        try:
            with ms_open(yaml_path, max_size=TENSOR_MAX_SIZE) as f:
                op_map = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f'{e}') from e

        return cls(op_map)

    def map_op(self, ori_op: str) -> str:
        return self._op_map.get(ori_op)

    def map_onnx_op(self, ori_op: str, opset_version: int) -> str:
        key = f'ai.onnx::{str(opset_version)}::{ori_op}'
        return self._op_map.get(key)
