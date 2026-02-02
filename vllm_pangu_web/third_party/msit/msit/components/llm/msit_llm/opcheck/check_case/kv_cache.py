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
import torch_npu

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class OpcheckKvCacheOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        # cache_in与cache_out只支持float16/int8数据类型
        newkv = in_tensors[0]
        layer_id = in_tensors[1]
        cache_in = in_tensors[2]
        token_offset = in_tensors[3]
        seqlen = in_tensors[4]
        cache_out = torch.zeros(shape=cache_in.shape).type(cache_in.dtype)
        batch = len(seqlen)

        prefix_ntokens = 0
        for i in range(batch):
            for j in range(seqlen[i]):
                cache_out[layer_id[0]][i][token_offset[i] - seqlen[i] + j][:] = newkv[prefix_ntokens + j][:]
            prefix_ntokens += seqlen[i]

        return [newkv, layer_id, cache_out, token_offset, seqlen]

    def test(self):
        self.execute_inplace()