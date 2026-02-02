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
import json

from typing import List

from model_evaluation.common import utils
from model_evaluation.common.enum import AtcErr, Framework
from components.utils.log import logger
from model_evaluation.parser.atc import AtcErrParser
from model_evaluation.bean import OpInfo, ConvertConfig
from components.utils.check.rule import Rule
from components.utils.file_open_check import ms_open
from components.utils.constants import TENSOR_MAX_SIZE


class ModelParser:
    def __init__(self, model_path: str, out_path: str, config: ConvertConfig) -> None:
        self._model_path = model_path
        self._out_path = out_path
        self._config = config

        self._json_path = ''
        self._om_path = ''

    def __del__(self):
        if os.path.isfile(self._json_path):
            try:
                os.remove(self._json_path)
            except Exception as e:
                logger.error(f"can not remove json file, please check.")
                raise RuntimeError from e
        if os.path.isfile(self._om_path):
            try:
                os.remove(self._om_path)
            except Exception as e:
                logger.error(f"can not remove om file, please check.")
                raise RuntimeError from e

    @property
    def om_path(self) -> str:
        return self._om_path

    def parse_all_ops(self, convert=False) -> List[OpInfo]:
        if self._json_path == '' and convert:
            if not self.parse_model_to_json():
                logger.error('parse model ops failed.')
                return []

        try:
            Rule.input_file().check(self._json_path, will_raise=True)
            with ms_open(self._json_path, max_size=TENSOR_MAX_SIZE) as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f'load ops json failed err:{e}')
            logger.error('It may be that your model is too large,'
                         'and the converted JSON is truncated '
                         'when you get content from ATC to JSON')
            return []

        nodes_attr_map = {Framework.ONNX: 'node', Framework.TF: 'node', Framework.CAFFE: 'layer'}

        framework = self._config.framework
        nodes_attr = nodes_attr_map.get(framework)
        nodes = data.get(nodes_attr)
        if not isinstance(nodes, list):
            logger.error(f'no attr \'{nodes_attr}\' in ops json.')
            return []

        op_type_attr_map = {Framework.ONNX: 'op_type', Framework.TF: 'op', Framework.CAFFE: 'type'}

        op_type_attr = op_type_attr_map.get(framework)
        op_infos: List[OpInfo] = []
        for node in nodes:
            ori_op_name = node.get('name')
            ori_op_type = node.get(op_type_attr)
            if not isinstance(ori_op_name, str) or not isinstance(ori_op_type, str):
                continue
            op_info = OpInfo(op_name=ori_op_name, op_type=ori_op_type)
            op_infos.append(op_info)
        return op_infos

    def parse_model_to_json(self) -> bool:
        model = os.path.basename(self._model_path)
        output = os.path.join(self._out_path, f'{model}.json')

        if not utils.check_file_security(output):
            return False
        if os.path.isfile(output):
            try:
                os.remove(output)
            except Exception as e:
                logger.error(f"can not remove output file, please check.")
                raise RuntimeError from e

        framework = self._config.framework
        convert_cmd = [
            'atc',
            '--mode=1',
            '--framework={}'.format(framework.value),
            '--om={}'.format(self._model_path),
            '--json={}'.format(output),
        ]
        logger.info('convert model to json, please wait...')
        out, err = utils.exec_command(convert_cmd)
        if len(err) != 0:
            logger.error(f'convert model to json failed, err:{err}.')
            return False

        errcode = AtcErrParser.parse_errcode(out)
        if errcode != AtcErr.SUCCESS:
            logger.error(f'convert model to json failed, err:{out}.')
            return False
        logger.info('convert model to json finished.')

        self._json_path = output
        return True

    def parse_model_to_om(self):
        model = os.path.basename(self._model_path)
        output = os.path.join(self._out_path, f'{model}')

        om_path = f'{output}.om'
        if not utils.check_file_security(om_path):
            return AtcErr.UNKNOWN, ''
        if os.path.isfile(om_path):
            try:
                os.remove(om_path)
            except Exception as e:
                logger.error(f"can not remove output om file, please check.")
                raise RuntimeError from e

        convert_cmd = [
            'atc',
            '--model={}'.format(self._model_path),
            '--framework={}'.format(self._config.framework.value),
            '--soc_version={}'.format(self._config.soc_type),
            '--output={}'.format(output),
        ]
        if self._config.framework == Framework.CAFFE:
            convert_cmd.append('--weight={}'.format(self._config.weight))
        logger.info('try to convert model to om, please wait...')
        out, err = utils.exec_command(convert_cmd)
        logger.info('try to convert model to om finished.')
        if len(err) != 0:
            return AtcErr.UNKNOWN, ''

        # parse and get error op name
        errcode = AtcErrParser.parse_errcode(out)
        if errcode != AtcErr.SUCCESS:
            return errcode, out

        self._om_path = om_path
        return errcode, ''
