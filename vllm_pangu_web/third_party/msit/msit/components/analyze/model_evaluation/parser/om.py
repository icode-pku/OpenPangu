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

from typing import List, Dict

from model_evaluation.common import utils
from components.utils.log import logger
from model_evaluation.common.enum import Engine, AtcErr
from model_evaluation.parser.atc import AtcErrParser
from model_evaluation.bean import OpInnerInfo
from components.utils.check.rule import Rule
from components.utils.file_open_check import ms_open
from components.utils.constants import TENSOR_MAX_SIZE


class OmParser:
    def __init__(self, om_path: str, out_path: str) -> None:
        self._om_path = om_path
        self._out_path = out_path

        self._om_json = ''

    def __del__(self):
        if os.path.isfile(self._om_json):
            os.remove(self._om_json)

    @staticmethod
    def _parse_fusion_op(attr_dict: Dict[str, Dict]) -> List[str]:
        attr = attr_dict.get('_datadump_original_op_names')
        if attr is None:
            return []
        try:
            ori_ops = attr.get('value').get('list').get('s')
        except AttributeError:
            return []
        if not isinstance(ori_ops, list):
            return []
        return ori_ops

    @staticmethod
    def _parse_op_engine(attr_dict: Dict[str, Dict]) -> str:
        attr = attr_dict.get('_ge_attr_op_kernel_lib_name')
        if attr is None:
            return Engine.UNKNOWN
        try:
            engine = attr.get('value').get('s')
        except AttributeError:
            return Engine.UNKNOWN
        if not isinstance(engine, str):
            return Engine.UNKNOWN

        engine_map = {'aicore': Engine.AICORE, 'aicpu': Engine.AICPU, 'dvpp': Engine.DVPP, 'cpu': Engine.HOST_CPU}
        engine = engine.lower()
        for key, value in engine_map.items():
            if key in engine:
                return value
        return Engine.UNKNOWN

    def parse_all_ops(self, convert=False) -> List[OpInnerInfo]:
        '''parse all op info from om json'''
        if len(self._om_json) == 0 and convert:
            if not self.parse_om_to_json():
                logger.error('parse model ops failed.')
                return []

        try:
            Rule.input_file().check(self._om_json, will_raise=True)
            with ms_open(self._om_json, max_size=TENSOR_MAX_SIZE) as f:
                om_data = json.load(f)
        except Exception as e:
            logger.error(f'load ops json failed, err:{e}')
            return []

        try:
            ops = om_data.get('graph')[0].get('op')
        except Exception as e:
            logger.error(f'ops data is invalid, err:{e}')
            return []

        if not isinstance(ops, list):
            logger.error('ops data is invalid.')
            return []

        op_infos: List[OpInnerInfo] = []
        for op_dict in ops:
            op_info = self._parse_op_info(op_dict)
            if op_info.op_name == '' or op_info.op_type == '':
                continue
            op_infos.append(op_info)
        return op_infos

    def parse_om_to_json(self) -> bool:
        model = os.path.basename(self._om_path)
        om_json = os.path.join(self._out_path, f'{model}.json')
        if not utils.check_file_security(om_json):
            return False
        if os.path.isfile(om_json):
            os.remove(om_json)

        convert_cmd = ['atc', '--mode=1', '--om={}'.format(self._om_path), '--json={}'.format(om_json)]
        logger.info('convert om to json, please wait...')
        out, err = utils.exec_command(convert_cmd)
        if len(err) != 0:
            logger.error(f'convert model to json failed, err:{err}.')
            return False

        errcode = AtcErrParser.parse_errcode(out)
        if errcode != AtcErr.SUCCESS:
            logger.error(f'convert model to json failed, err:{out}.')
            return False
        logger.info('convert om to json finished.')

        self._om_json = om_json
        return True

    def _parse_op_info(self, op_dict: Dict) -> OpInnerInfo:
        '''parse single op info from om json'''
        op_info = OpInnerInfo()

        op_name = op_dict.get('name')
        op_type = op_dict.get('type')
        if not isinstance(op_name, str) or not isinstance(op_type, str):
            return op_info
        op_info.op_name = op_name
        op_info.op_type = op_type

        # parse operator attr
        attrs = op_dict.get('attr')
        if not isinstance(attrs, list):
            return op_info
        attr_dict: Dict[str, Dict] = {}
        for attr in attrs:
            key = attr.get('key')
            if not isinstance(key, str):
                continue
            attr_dict[key] = attr

        # parse fusion operators
        ori_ops = self._parse_fusion_op(attr_dict)
        if len(ori_ops) > 1:
            op_info.is_fusion = True
            op_info.ori_ops = ori_ops
        else:
            op_info.ori_ops = [op_name]

        # parse operator engine
        engine = self._parse_op_engine(attr_dict)
        if isinstance(engine, Engine):
            op_info.op_engine = engine

        return op_info
