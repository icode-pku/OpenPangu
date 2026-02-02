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
import csv

from copy import deepcopy
from typing import Dict

from model_evaluation.common import utils
from components.utils.log import logger
from model_evaluation.common import Const
from model_evaluation.common.enum import Engine
from components.utils.file_open_check import ms_open

OP_FILTER_LIST = ['Constant', 'Const', 'Input', 'Placeholder']


class OpResult:
    '''Operator analysis result'''

    def __init__(self, ori_op_name, ori_op_type, soc_type='', is_supported=True, details=''):  # origin op type
        self._ori_op_name = ori_op_name
        self._ori_op_type = ori_op_type
        self._op_name = ''
        self._op_type = ''
        self._op_engine = Engine.UNKNOWN
        self._soc_type = soc_type
        self._is_supported = is_supported
        self._details = details

    @property
    def ori_op_name(self):
        return self._ori_op_name

    @ori_op_name.setter
    def ori_op_name(self, ori_op_name_):
        self._ori_op_name = ori_op_name_

    @property
    def ori_op_type(self):
        return self._ori_op_type

    @ori_op_type.setter
    def ori_op_type(self, ori_op_type_):
        self._ori_op_type = ori_op_type_

    @property
    def op_name(self):
        return self._op_name

    @op_name.setter
    def op_name(self, op_name_):
        self._op_name = op_name_

    @property
    def op_type(self):
        return self._op_type

    @op_type.setter
    def op_type(self, op_type_):
        self._op_type = op_type_

    @property
    def op_engine(self):
        return self._op_engine

    @op_engine.setter
    def op_engine(self, op_engine_):
        self._op_engine = op_engine_

    @property
    def soc_type(self):
        return self._soc_type

    @soc_type.setter
    def soc_type(self, soc_type_):
        self._soc_type = soc_type_

    @property
    def is_supported(self):
        return self._is_supported

    @is_supported.setter
    def is_supported(self, is_supported_):
        self._is_supported = is_supported_

    @property
    def details(self):
        return self._details

    def set_details(self, err_detail: str) -> None:
        if len(self._details) != 0:
            if err_detail not in self._details.split(';'):
                self._details += ';' + err_detail
        else:
            self._details = err_detail


class Result:
    def __init__(self) -> None:
        self._op_results: Dict[str, OpResult] = {}

    def insert(self, op_result: OpResult) -> None:
        ori_op = op_result.ori_op_name
        if isinstance(ori_op, str):
            self._op_results[ori_op] = deepcopy(op_result)

    def get(self, ori_op: str) -> OpResult:
        return self._op_results.get(ori_op)

    def dump(self, out_path: str):
        out_csv = os.path.join(out_path, 'result.csv')
        if not utils.check_file_security(out_csv):
            return
        if os.path.isfile(out_csv):
            os.remove(out_csv)
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            modes = os.st.S_IRUSR | os.st.S_IWUSR | os.st.S_IRGRP
            f = ms_open(out_csv, 'w', newline='')
        except Exception as e:
            logger.error(f'open result.csv failed, err:{e}')
        fields = ['ori_op_name', 'ori_op_type', 'op_name', 'op_type', 'soc_type', 'engine', 'is_supported', 'details']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        err_op_num = 0
        for op_result in self._op_results.values():
            if op_result.ori_op_type in OP_FILTER_LIST:
                continue
            row = {
                'ori_op_name': op_result.ori_op_name,
                'ori_op_type': op_result.ori_op_type,
                'op_name': op_result.op_name,
                'op_type': op_result.op_type,
                'soc_type': op_result.soc_type,
                'engine': op_result.op_engine.name,
                'is_supported': op_result.is_supported,
                'details': op_result.details,
            }
            writer.writerow(row)
            if not op_result.is_supported:
                err_op_num += 1
        f.flush()
        f.close()
        os.chmod(out_csv, Const.ONLY_READ)
        logger.info(f'analysis result has bean written in {out_csv}.')
        logger.info(f'number of abnormal operators: {err_op_num}.')
