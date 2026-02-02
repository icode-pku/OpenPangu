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

from msit_llm.opcheck import operation_test
from components.utils.util import safe_get


class OpcheckSetValueOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        starts = self.op_param.get("starts", None)
        ends = self.op_param.get("ends", None)
        strides = self.op_param.get("strides", None)
        golden_result = [safe_get(in_tensors, 0).clone(), safe_get(in_tensors, 1).clone()]
        for i, _ in enumerate(starts):
            self.validate_int_range(strides[i], [1], "strides") # 当前仅支持strides为全1
            start = safe_get(starts, i)
            end = safe_get(ends, i)
            stride = safe_get(strides, i)
            golden_result[0][start:end:stride].copy_(safe_get(in_tensors, 1))
        return golden_result

    def test(self):
        ret = self.validate_param("starts", "ends", "strides")
        if not ret:
            return
        self.execute()