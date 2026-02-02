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

from enum import unique, Enum
from components.utils.parser import BaseCommand
from components.utils.log import logger
from model_convert.cmd_utils import add_arguments, gen_convert_cmd, execute_cmd


@unique
class SocType(Enum):
    Ascend310 = 0
    Ascend310P = 1


def get_soc_type() -> str:
    default_soc = SocType.Ascend310.name
    try:
        import acl

        return acl.get_soc_name()
    except ImportError:
        logger.warning(f'Get soc_version failed, use default {default_soc}.')
    return default_soc


class ModelConvertCommand(BaseCommand):
    def __init__(self, backend, *args, **kwargs):
        super(ModelConvertCommand, self).__init__(*args, **kwargs)
        self.conf_args = None
        self.backend = backend

    def add_arguments(self, parser, **kwargs):
        self.conf_args = add_arguments(parser, backend=self.backend)

    def handle(self, args, **kwargs):
        convert_cmd = gen_convert_cmd(self.conf_args, args, backend=self.backend)
        execute_cmd(convert_cmd)


def get_cmd_instance():
    atc_cmd = ModelConvertCommand(name="atc", help_info="Convert onnx to om by atc.", backend="atc")
    aoe_cmd = ModelConvertCommand(name="aoe", help_info="Convert onnx to om by aoe.", backend="aoe")
    convert_cmd = BaseCommand(
        name="convert",
        help_info="convert tool converts the model from ONNX, TensorFlow, Caffe and MindSpore to OM. \
                   It supports atc, aoe.",
        children=[atc_cmd, aoe_cmd]
    )
    return convert_cmd
