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
import importlib

from auto_optimizer.common.utils import format_to_module
from components.debug.common import logger


class Register:
    def __init__(self, path_name: str):
        """
        path_name: 待注册文件夹的绝对路径
        """
        try:
            real_path = os.path.realpath(path_name)
        except Exception as err:
            raise RuntimeError("Invalid file error={}".format(err)) from err
        self.path_name = real_path

    @staticmethod
    def import_module(module):
        errors = []
        try:
            return importlib.import_module(module)
        except ImportError as error:
            errors.append((module, error))

        Register._handle_errors(errors)
        return None

    @staticmethod
    def _handle_errors(errors):
        if not errors:
            return

        for name, err in errors:
            raise RuntimeError("Module {} import failed: {}".format(name, err))

    def import_modules(self):
        modules = []
        try:
            self._add_modules(modules)
        except Exception as error:
            logger.error("add_modules failed, {}".format(error))
            raise RuntimeError("add_modules failed: {}".format(error)) from error

        for module in modules:
            if not module:
                continue
            Register.import_module(module)
        return True

    def _add_modules(self, modules: list):
        pwd_dir = os.getcwd()

        for root, _, files in os.walk(self.path_name, topdown=False):
            modules += [format_to_module(os.path.join(root.split(pwd_dir)[1], file)) for file in files]
