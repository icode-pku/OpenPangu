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

from typing import Dict

from auto_optimizer.common.register import Register
from auto_optimizer.common.utils import format_to_module


class ConfigDict(Dict):
    def __missing__(self, key):
        raise KeyError(key)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as err:
            raise AttributeError(f"'ConfigDict' object has no attribute '{item}'") from err


class Config:
    def __init__(self, config_dict=None):
        config_dict = config_dict or {}
        if not isinstance(config_dict, dict):
            raise TypeError('config_dict must be dict')

        super().__setattr__('_config_dict', ConfigDict(config_dict))

    def __repr__(self):
        return f'{self._config_dict.__repr__()}'

    def __len__(self):
        return len(self._config_dict)

    def __getattr__(self, item):
        return getattr(self._config_dict, item)

    def __getitem__(self, item):
        return self._config_dict[item]

    def __iter__(self):
        return iter(self._config_dict)

    def __getstate__(self):
        return self._config_dict

    @staticmethod
    def read_by_file(file_name):
        """
        读取模型配置文件，返回模型相关配置参数及推理流程
        """
        format_path = format_to_module(file_name)
        try:
            model_dict = Register.import_module(format_path)
        except Exception as err:
            raise RuntimeError("invalid read file error={}".format(err)) from err
        if not isinstance(model_dict.model, Dict):
            raise RuntimeError("config is not Dict")

        config_dict = model_dict.model

        return Config(config_dict)
