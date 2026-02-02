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

from enum import Enum, unique

from model_evaluation.common.enum.atc_err import AtcErr


@unique
class Rule(Enum):
    EVAL_ATC_SUCCESS = 0
    EVAL_ATC_UNSUPPORT_OP_ERR = 1
    EVAL_ATC_OTHER_ERR = 2

    @staticmethod
    def get_rule_with_atc_err(err: AtcErr):
        if err == AtcErr.SUCCESS:
            return Rule.EVAL_ATC_SUCCESS
        unsupported_op_err = [AtcErr.E19010, AtcErr.EZ0501, AtcErr.EZ3002, AtcErr.EZ3003, AtcErr.EZ9010]
        if err in unsupported_op_err:
            return Rule.EVAL_ATC_UNSUPPORT_OP_ERR

        return Rule.EVAL_ATC_OTHER_ERR
