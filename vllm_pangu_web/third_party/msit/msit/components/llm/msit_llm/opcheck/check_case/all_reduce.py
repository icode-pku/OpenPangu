# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import torch

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class OpcheckAllReduceOperation(operation_test.OperationTest):
    @staticmethod
    def sum_cal(in_tensors):
        result = in_tensors[0].clone()
        for i in range(1, len(in_tensors)):
            result += in_tensors[i]
        return [result]

    @staticmethod
    def max_cal(in_tensors):
        result = in_tensors[0].clone()
        for i in range(1, len(in_tensors)): 
            result = torch.max(result, in_tensors[i])
        return [result]

    @staticmethod
    def min_cal(in_tensors):
        result = in_tensors[0].clone()
        for i in range(1, len(in_tensors)): 
            result = torch.min(result, in_tensors[i])
        return [result]

    @staticmethod
    def prod_cal(in_tensors):
        result = in_tensors[0].clone()
        for i in range(1, len(in_tensors)):
            result = torch.mul(result, in_tensors[i])
        return [result]

    def golden_calc(self, in_tensors):
        all_reduce_type = self.op_param.get('allReduceType', None)
        new_in_tensors = self.get_new_in_tensors()
                    
        if all_reduce_type == "sum":
            golden = OpcheckAllReduceOperation.sum_cal(new_in_tensors)
        elif all_reduce_type == "max":
            golden = OpcheckAllReduceOperation.max_cal(new_in_tensors)
        elif all_reduce_type == "min":
            golden = OpcheckAllReduceOperation.min_cal(new_in_tensors)
        elif all_reduce_type == "prod":
            golden = OpcheckAllReduceOperation.prod_cal(new_in_tensors)

        return golden

    def test_all_reduce(self):
        if self.pid is None:
            logger_text = "Cannot get a valid pid, AllReduceOperation is not supported!"
            logger.error(logger_text)
            return

        ret = self.validate_param("allReduceType", "rank", "rankRoot", "rankSize")
        if not ret:
            return

        all_reduce_type = self.op_param.get('allReduceType', "sum")
        backend = self.op_param.get('backend', "hccl")
        logger_text1 = f"backend: {backend}, allreduceType: {all_reduce_type}"
        logger_text2 = "env: {}".format(os.getenv("LCCL_DETERMINISTIC", ""))
        logger_text3 = "env: {}".format(os.getenv("HCCL_DETERMINISTIC", ""))
        logger.debug(logger_text1)
        logger.debug(logger_text2)
        logger.debug(logger_text3)

        self.execute()