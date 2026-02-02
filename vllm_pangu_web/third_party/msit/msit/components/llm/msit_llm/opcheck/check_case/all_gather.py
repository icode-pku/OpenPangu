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

import torch

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class OpcheckAllGatherOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        new_in_tensors = self.get_new_in_tensors()
        golden_result = torch.stack(new_in_tensors, dim=0)
        return [golden_result]        

    def test_all_gather(self):
        if self.pid is None:
            logger_text = "Cannot get a valid pid, AllGatherOperation is not supported!"
            logger.error(logger_text)
            return

        ret = self.validate_param("rank", "rankRoot", "rankSize")
        if not ret:
            return

        self.execute()