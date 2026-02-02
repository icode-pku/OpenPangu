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

from enum import Enum
from ctypes import CDLL
import torch

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class TopkToppSamplingType(Enum):
    SAMPLING_UNDEFINED = -1 # 未定义
    SINGLE_TOPK_SAMPLING = 0 # 非batch级别随机种子、Topk的取样
    BATCH_TOPK_MULTINOMIAL_SAMPLING = 1 # batch级别随机种子、Topk的multinomial取样
    BATCH_TOPK_EXPONENTIAL_SAMPLING = 2 # batch级别随机种子、Topk的exponential取样
    SAMPLING_MAX = 3 # 取样类型最大值（暂不支持）


class OpcheckToppOperation(operation_test.OperationTest):
    def single_topk_sampling(self, in_tensors):
        libc = CDLL("libc.so.6")
        topk = self.op_param.get('topk', 100)
        rand_seed = self.op_param.get('randSeed', 0)
        logger_text = f"topk: {topk}, rand_seed: {rand_seed}"
        logger.debug(logger_text)

        libc.srand(rand_seed)
        rand_list = [libc.rand() / 0x7fffffff for i in range(64)]

        probs = in_tensors[0]
        topp = in_tensors[1]
        probs_sorted = torch.sort(probs, axis=-1)[..., ::-1][..., :topk]
        indices_sorted = torch.argsort(-probs, kind='mergesort', axis=-1)[..., :topk]
        probs_sorted_sumed = torch.cumsum(probs_sorted, axis=-1)
        bool_judge = (probs_sorted_sumed < topp)
        sum_val = torch.sum(bool_judge, axis=-1, keepdims=True) - 1
        sum_val[sum_val < 0] = 0
        topp_v = torch.take_along_axis(probs_sorted_sumed, sum_val, axis=-1)
        if probs.shape[0] > len(rand_list):
            raise ValueError(f"size of probs ({probs.shape[0]}) exceeds length of rand_list ({len(rand_list)})")
        topp_v *= torch.tensor(rand_list).reshape(-1, 1)[0:probs.shape[0]]
        bool_judge_one = probs_sorted_sumed < topp_v
        res = torch.sum(bool_judge_one, axis=-1, keepdims=True)
        res[res < 0] = 0
        indices_sampled = torch.take_along_axis(indices_sorted, res, axis=-1)
        probs_sampled = torch.take_along_axis(probs_sorted, res, axis=-1)
        golden_out = [indices_sampled.type(torch.int32), probs_sampled.type(torch.float16)]
        return golden_out

    def sampling_undefined(self, in_tensors, topktopp_sampling_type):
        libc = CDLL("libc.so.6")
        probs = in_tensors[0]
        topk = in_tensors[1]
        topp = in_tensors[2]
        probs_sorted, idx_sorted = torch.sort(probs, descending=True, stable=True)
        gather_topk = torch.gather(probs_sorted, dim=1, index=topk)
        topk_mask = torch.lt(probs_sorted, gather_topk).to(torch.bool)
        probs_masked_topk = probs_sorted.masked_fill(topk_mask, 0)
        probs_cumsumed = torch.cumsum(probs_masked_topk, axis=-1)
        if topktopp_sampling_type == TopkToppSamplingType.BATCH_TOPK_EXPONENTIAL_SAMPLING.value:
            exp = in_tensors[3]
            topp_mask = torch.gt(probs_cumsumed, topp).to(torch.bool)
            probs_masked_topp = probs_cumsumed.masked_fill(topp_mask, 0)
            divided_probs = torch.div(probs_masked_topp, exp)
            argmax_probs, argmax_idx = torch.sort(divided_probs, descending=True, stable=True)
            argmax_probs = argmax_probs[..., :1]
            argmax_idx = argmax_idx[..., :1]
            outtensor_probs = torch.gather(probs_sorted, dim=1, index=argmax_idx)
            outtensor_idx = torch.gather(idx_sorted, dim=1, index=argmax_idx)
            golden_out = [outtensor_idx.type(torch.int32), outtensor_probs.type(torch.float16)]
        if topktopp_sampling_type == TopkToppSamplingType.BATCH_TOPK_MULTINOMIAL_SAMPLING.value: 
            rand_seeds = self.op_param.get('randSeeds', None)
            bool_judge = (probs_cumsumed < topp)
            sum_val = torch.sum(bool_judge, axis=-1, keepdims=True) - 1
            sum_val[sum_val < 0] = 0
            topp_v = torch.take_along_axis(probs_cumsumed, sum_val, axis=-1)
            randnp_new = torch.zeros(probs_cumsumed.shape[0])
            for i in range(probs_cumsumed.shape[0]):
                libc.srand(rand_seeds[i])
                rand_num = libc.rand() / 0x7fffffff
                randnp_new[i] = rand_num
            randnp_new = randnp_new.reshape(-1, 1)
            topp_v_new = randnp_new[0:probs_cumsumed.shape[0]] * topp_v
            bool_judge_one = (probs_cumsumed <= topp_v_new)
            res = torch.sum(bool_judge_one, axis=-1, keepdims=True)
            res[res < 0] = 0
            outtensor_probs = torch.gather(probs_sorted, dim=1, index=res)
            outtensor_idx = torch.gather(idx_sorted, dim=1, index=res)

            rand_list = torch.rand(probs.shape[0], device=probs.device)
            probs_sorted_sumed = torch.cumsum(probs_sorted, axis=-1)
            golden_out = [outtensor_idx.type(torch.int32), outtensor_probs.type(torch.float16)]
        return golden_out


    def golden_calc(self, in_tensors):
        topktopp_sampling_type = self.op_param.get('topktopp_sampling_type', 
                                                   TopkToppSamplingType.SAMPLING_UNDEFINED.value)

        if topktopp_sampling_type == TopkToppSamplingType.SINGLE_TOPK_SAMPLING.value:
            golden_out = self.single_topk_sampling(in_tensors)
        else:
            golden_out = self.sampling_undefined(in_tensors, topktopp_sampling_type)
        return golden_out

    def test(self):
        ret = self.validate_param("tokkToppSamplingType")
        if not ret:
            return
        self.execute()