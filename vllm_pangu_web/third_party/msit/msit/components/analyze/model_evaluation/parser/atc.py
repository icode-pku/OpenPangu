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
import re

from typing import Dict, Set, Any

from model_evaluation.common.enum import AtcErr


OP_PATTERN = '[A-Za-z0-9-_/.:]{0,100}'

EZ0501_PATTERN = f'IR for Op \[{OP_PATTERN}, optype \[{OP_PATTERN}\]\], is not registered.'
EZ3002_PATTERN = f'Optype \[{OP_PATTERN}\] of Ops kernel \[{OP_PATTERN}\] is unsupported.'
EZ3003_PATTERN = f'No supported Ops kernel and engine are found for \[{OP_PATTERN}\], optype \[{OP_PATTERN}\].'
E_9010_PATTERN = f'No parser is registered for Op \[{OP_PATTERN}, optype \[{OP_PATTERN}\]\].'


class AtcErrParser:
    @classmethod
    def parse_errcode(cls, errinfo: str) -> AtcErr:
        if errinfo.find('ATC run success') != -1:
            return AtcErr.SUCCESS
        # parse error code
        for k, v in AtcErr.__members__.items():
            if errinfo.find(k) != -1:
                return v
        return AtcErr.UNKNOWN

    @classmethod
    def parse_unsupported_op(cls, errinfo: str) -> Any:
        errcode = cls.parse_errcode(errinfo)

        err_parser_map = {
            AtcErr.EZ3003: cls.parse_err_ez3003,
            AtcErr.E19010: cls.parse_err_e_9010,
            AtcErr.EZ9010: cls.parse_err_e_9010,
            AtcErr.EZ0501: cls.parse_err_ez0501,
            AtcErr.EZ3002: cls.parse_err_ez3002,
        }

        if errcode in err_parser_map:
            return err_parser_map.get(errcode)(errinfo)

        return {}

    @classmethod
    def parse_err_ez0501(cls, errinfo: str) -> Dict[str, str]:
        op_infos: Dict[str, str] = {}
        lines = errinfo.split(os.linesep)
        for line in lines:
            matcher = re.search(EZ0501_PATTERN, line)
            if matcher is None:
                continue
            info = matcher.group()
            left = info.find('[')
            right = info.find(',', left)
            op_name = info[left + 1: right]
            left = info.find('[', right)
            right = info.find(']', left)
            op_type = info[left + 1: right]
            op_infos[op_name] = op_type
        return op_infos

    @classmethod
    def parse_err_ez3002(cls, errinfo: str) -> Set[str]:
        op_infos: Set[str] = set()
        lines = errinfo.split(os.linesep)
        for line in lines:
            matcher = re.search(EZ3002_PATTERN, line)
            if matcher is None:
                continue
            info = matcher.group()
            left = info.find('[')
            right = info.find(']', left)
            op_type = info[left + 1: right]
            op_infos.add(op_type)
        return op_infos

    @classmethod
    def parse_err_ez3003(cls, errinfo: str) -> Dict[str, str]:
        op_infos: Dict[str, str] = {}
        lines = errinfo.split(os.linesep)
        for line in lines:
            matcher = re.search(EZ3003_PATTERN, line)
            if matcher is None:
                continue
            info = matcher.group()
            left = info.find('[')
            right = info.find(']', left)
            op_name = info[left + 1: right]
            left = info.find('[', right)
            right = info.find(']', left)
            op_type = info[left + 1: right]
            op_infos[op_name] = op_type
        return op_infos

    @classmethod
    def parse_err_e_9010(cls, errinfo: str) -> Dict[str, str]:
        op_infos: Dict[str, str] = {}
        lines = errinfo.split(os.linesep)
        for line in lines:
            matcher = re.search(E_9010_PATTERN, line)
            if matcher is None:
                continue
            info = matcher.group()
            left = info.find('[')
            right = info.find(',', left)
            op_name = info[left + 1: right]
            left = info.find('[', right)
            right = info.find(']', left)
            op_type = info[left + 1: right]
            op_infos[op_name] = op_type
        return op_infos
