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
import json
import math
import torch
import torch_npu
import numpy as np

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class CalcType(Enum):
    UNDEFINED = 0 # decoder&encoder for flashAttention
    ENCODER = 1 # encoder for flashAttention
    DECODER = 2 # decoder for flashAttention
    PA_ENCODER = 3 # encoder for pagedAttention


class KvCacheCfg(Enum):
    K_CACHE_V_CACHE = 0 # 默认值，进行kvcache处理
    K_BYPASS_V_BYPASS = 1 # 直接传入kvcache


class MaskType(Enum):
    MASK_TYPE_UNDEFINED = 0 # 默认值，全0mask
    MASK_TYPE_NORM = 1 # 倒三角mask
    MASK_TYPE_ALIBI = 2 # alibi mask
    MASK_TYPE_NORM_COMPRESS = 3 # 倒三角压缩mask
    MASK_TYPE_ALIBI_COMPRESS = 4 # alibi压缩mask
    MASK_TYPE_ALIBI_COMPRESS_SQRT = 5 # alibi压缩开平方mask
    MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN = 6 # alibi压缩mask左对齐，只支持Atlas 800I A2


class KernelType(Enum):
    KERNELTYPE_DEFAULT = 0 # i:fp16, bmm:fp16, o:fp16
    KERNELTYPE_HIGH_PRECISION = 1 # i:fp16, bmm:fp32, o:fp16


class ClampType(Enum):
    CLAMP_TYPE_UNDEFINED = 0 # 不做clamp
    CLAMP_TYPE_MIN_MAX = 1 # 做clamp，同时指定最大最小值


class OpcheckUnpadSelfAttentionOperation(operation_test.OperationTest):
    @staticmethod
    def group_matmul(head_num, group_num, in_a, in_b):
        try:
            group_head = head_num // group_num
        except ZeroDivisionError as e:
            raise RuntimeError("get ZeroDivisionError when calc SelfAttentionOperation golden") from e     
        score = None
        for i in range(group_num):
            group_score = torch.matmul(
                in_a.to(torch.bfloat16)[i * group_head: (i + 1) * group_head, :, :], 
                in_b.to(torch.bfloat16)[i:(i + 1), :, :]
            )
            if score is None:
                score = group_score
            else:
                score = torch.concat((score, group_score), 0)
        logger.debug(score.shape)
        return score

    @staticmethod
    def attention_mask_nz_2_nd(attention_mask, seq_len):
        attention_mask = attention_mask.contiguous().permute(0, 2, 1, 3)
        dim0, dim1, dim2, dim3 = attention_mask.shape
        attention_mask = attention_mask.contiguous().view(dim0 * dim1, dim2, dim3)
        attention_mask = attention_mask[:, :, :dim1]
        batch = len(seq_len)
        if dim0 == 1:
            attention_mask = attention_mask.contiguous().view(dim1, dim1)
        elif dim0 != batch:
            attention_mask = attention_mask.contiguous().view(batch, dim0 // batch, dim1, dim1)
        return attention_mask

    @staticmethod
    def reshape_qkv(qkv, is_pa=False):
        if len(qkv.shape) == 4:
            dim0, dim1, dim2, dim3 = qkv.shape
            qkv = qkv.contiguous().view(dim0 * dim1, dim2 * dim3)
        if is_pa and len(qkv.shape) == 3:
            dim0, dim1, dim2 = qkv.shape
            qkv = qkv.contiguous().view(dim0, dim1 * dim2)
        return qkv

    @staticmethod
    def get_qkv(in_tensors):
        q, k, v = in_tensors[:3]
        q = OpcheckUnpadSelfAttentionOperation.reshape_qkv(q, True)
        k = OpcheckUnpadSelfAttentionOperation.reshape_qkv(k, True)
        v = OpcheckUnpadSelfAttentionOperation.reshape_qkv(v, True)
        return q, k, v

    @staticmethod
    def get_out_sub(head_info, q_s, score, v_slice, _p):
        head_num, head_size, kv_head_num = head_info
        score_max = torch.max(score, axis=-1).values
        score = score - score_max.view((head_num, q_s, 1))
        score_exp = torch.exp(score)
        score_sum = torch.sum(score_exp, axis=-1)

        _p = score_exp.view([-1, ]) if _p is None else torch.concat((_p, score_exp.view([-1, ])), 0)
        p = score_exp / score_sum.view((head_num, q_s, 1))

        out_sub = OpcheckUnpadSelfAttentionOperation.group_matmul(head_num, kv_head_num, p, v_slice)
        out_sub = out_sub.view([head_num, q_s, head_size]).permute(1, 0, 2).contiguous()
        return out_sub, _p

    def get_mask(self, in_tensors, seq_len):
        mask = in_tensors[3]
        soc_version = self.get_soc_version()
        if soc_version != "Ascend910B":
            mask = OpcheckUnpadSelfAttentionOperation.attention_mask_nz_2_nd(mask, seq_len)
            mask_type = self.op_param.get("maskType", MaskType.MASK_TYPE_UNDEFINED.value)
            if mask_type == MaskType.MASK_TYPE_ALIBI.value and len(mask.shape) == 4:
                self.in_tensors[3] = mask.squeeze(dim=0)
            else:
                self.in_tensors[3] = mask
        if len(mask.shape) == 2:
            dim0, dim1 = mask.shape
            mask = mask.view(1, dim0, dim1)
        return mask

    def get_batch_status(self, in_tensors, seq_len):
        batch_run_status_enable = self.op_param.get("batchRunStatusEnable", False)
        if batch_run_status_enable:
            batch_status = in_tensors[-1]
            self.bind_idx.append(len(in_tensors) - 1)
        else:
            batch_status = range(len(seq_len))
        return batch_status

    def get_post_mask_coff(self, data_type):
        kernel_type = self.op_param.get("kernelType", KernelType.KERNELTYPE_DEFAULT.value)
        mask_type = self.op_param.get("maskType", MaskType.MASK_TYPE_UNDEFINED.value)
        is_alibi = mask_type == MaskType.MASK_TYPE_ALIBI.value or mask_type == MaskType.MASK_TYPE_ALIBI_COMPRESS.value \
            or mask_type == MaskType.MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN.value \
            or mask_type == MaskType.MASK_TYPE_ALIBI_COMPRESS_SQRT.value
        if kernel_type == KernelType.KERNELTYPE_HIGH_PRECISION.value or self.optimization_closed == "kernelType":
            post_mask_coff = 1.0
        elif data_type == torch.float16:
            post_mask_coff = 1.0
        elif (data_type == torch.bfloat16 or data_type == torch.float32) and is_alibi:
            post_mask_coff = 1.0
        else:
            post_mask_coff = -3e38
        return post_mask_coff

    def get_attention_params(self, q):
        q_scale = self.op_param.get("qScale", 1.0)
        qk_scale = self.op_param.get("qkScale", 1.0)
        head_num = self.op_param.get("headNum", 1)
        if head_num == 0:
            raise ZeroDivisionError("Parameter headNum is invalid, Please check!")
        head_size = int(q.shape[1] / head_num)
        kv_head_num = self.op_param.get("kvHeadNum", 0)
        data_type = q.dtype
        q_ntokens = q.shape[0]

        head_info = [head_num, head_size, kv_head_num]
        params = [q_scale, qk_scale, head_info, data_type, q_ntokens]
        return params

    def get_clamped_score(self, score):
        clamp_type = self.op_param.get("clampType", ClampType.CLAMP_TYPE_UNDEFINED.value)
        if clamp_type == ClampType.CLAMP_TYPE_MIN_MAX.value:
            clamp_min, clamp_max = self.op_param.get("clampMin", 0.0), self.op_param.get("clampMax", 0.0)
            clamp_min_brc = torch.ones(score.shape) * clamp_min
            clamp_max_brc = torch.ones(score.shape) * clamp_max
            score = torch.max(score, clamp_min_brc)
            score = torch.min(score, clamp_max_brc)
        return score

    def kv_bypass_golden_func(self, in_tensors):
        q, k, v = in_tensors[:3]
        mask = None
        seq_len = None
        mask_type = self.op_param.get("maskType", MaskType.MASK_TYPE_UNDEFINED.value)
        is_triu_mask = self.op_param.get("isTriuMask", 0)
        is_mask = mask_type != MaskType.MASK_TYPE_UNDEFINED.value
        if is_mask or self.optimization_closed == "maskType":
            kv_seqlen = in_tensors[4]
            seq_len = q_seqlen = in_tensors[5]
            layer_id = int(in_tensors[6][0])
            mask = self.get_mask(in_tensors, seq_len)
            if self.optimization_closed == "maskType":
                del self.in_tensors[3] 
        else:
            kv_seqlen = in_tensors[3]
            seq_len = q_seqlen = in_tensors[4]
            layer_id = int(in_tensors[5][0])

        batch_status = self.get_batch_status(in_tensors, seq_len)
        post_mask_coff = self.get_post_mask_coff(q.dtype)
        _, _, head_info, data_type, q_ntokens = self.get_attention_params(q)
        head_num, head_size, kv_head_num = head_info
        max_seq_len = max(seq_len)

        q_offset, k_offset, v_offset = 0, 0, 0
        s, _p, out = None, None, None

        for idx in batch_status:
            q_s, kv_s = q_seqlen[idx], kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:].view(q_s, head_num, head_size).permute(1, 0, 2)
            k_slice_t = k[layer_id][idx][:kv_s][:].view(kv_s, kv_head_num, head_size).permute(1, 2, 0) # get K^T
            v_slice = v[layer_id][idx][:kv_s][:].view(kv_s, kv_head_num, head_size).permute(1, 0, 2)

            score = OpcheckUnpadSelfAttentionOperation.group_matmul(head_num, kv_head_num, q_slice, k_slice_t)
            s = score.view([-1, ]) if s is None else torch.concat((s, score.view([-1, ])), 0)

            tor = 1.0 / math.sqrt(1.0 * head_size)
            score = score * tor
            score = self.get_clamped_score(score)
            if is_mask:
                score = score + mask[idx, :q_s, :kv_s] * post_mask_coff
            
            out_sub, _p = OpcheckUnpadSelfAttentionOperation.get_out_sub(head_info, q_s, score, v_slice, _p)
            out = out_sub if out is None else torch.concat((out, out_sub), 0)

            q_offset += q_s
            k_offset += max_seq_len
            v_offset += max_seq_len

        out = out.view(q_ntokens, head_num * head_size)
        return out.type(data_type)

    def pa_encoder_golden_func(self, in_tensors):
        q, k, v = OpcheckUnpadSelfAttentionOperation.get_qkv(in_tensors)
        mask, seq_len = None, None
        mask_type = self.op_param.get("maskType", MaskType.MASK_TYPE_UNDEFINED.value)
        is_mask = mask_type != MaskType.MASK_TYPE_UNDEFINED.value
        is_triu_mask = self.op_param.get("isTriuMask", 0)
        if is_mask or self.optimization_closed == "maskType":
            seq_len = in_tensors[4]
            mask = self.get_mask(in_tensors, seq_len)
            if self.optimization_closed == "maskType":
                del self.in_tensors[3]
                self.bind_idx.append(3)
            else:
                self.bind_idx.append(4)
        else:
            seq_len = in_tensors[3]
            self.bind_idx.append(3)

        batch_status = self.get_batch_status(in_tensors, seq_len)
        post_mask_coff = self.get_post_mask_coff(q.dtype)
        _, qk_scale, head_info, data_type, q_ntokens = self.get_attention_params(q)
        head_num, head_size, kv_head_num = head_info
        max_seq_len = max(seq_len)

        q_seqlen = kv_seqlen = seq_len # crossattention时，q_seqlen != k_seqlen
        q_offset, k_offset, v_offset = 0, 0, 0
        s, _p, out = None, None, None

        for idx in batch_status:
            q_s, kv_s = q_seqlen[idx], kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:].view(q_s, head_num, head_size).permute(1, 0, 2)
            k_slice_t = k[k_offset:k_offset + kv_s][:].view(kv_s, kv_head_num, head_size).permute(1, 2, 0)
            v_slice = v[v_offset:v_offset + kv_s][:].view(kv_s, kv_head_num, head_size).permute(1, 0, 2)

            score = OpcheckUnpadSelfAttentionOperation.group_matmul(head_num, kv_head_num, q_slice, k_slice_t)
            s = score.view([-1, ]) if s is None else torch.concat((s, score.view([-1, ])), 0)

            tor = qk_scale
            score = score * tor
            score = self.get_clamped_score(score)
            if is_mask:
                if (mask_type == MaskType.MASK_TYPE_NORM.value or mask_type == MaskType.MASK_TYPE_NORM_COMPRESS.value) \
                    and q_s > mask.shape[1]:
                    # 压缩norm mask, 使用当前最大seqlen生成mask
                    no_compress_mask = torch.ones(size=(1, max_seq_len, max_seq_len)).to(score.device)
                    no_compress_mask = torch.triu(no_compress_mask, 1)
                    no_compress_mask *= -10000.0
                    self.in_tensors[3] = no_compress_mask.to(self.in_tensors[3].dtype)
                    score = score + no_compress_mask[:, :q_s, :kv_s]
                else:
                    if len(mask.shape) == 4:
                        mask_cur = mask[idx]
                    else:
                        mask_cur = mask
                    score = score + mask_cur[:, :q_s, :kv_s] * post_mask_coff

            out_sub, _p = OpcheckUnpadSelfAttentionOperation.get_out_sub(head_info, q_s, score, v_slice, _p)
            out = out_sub if out is None else torch.concat((out, out_sub), 0)

            q_offset += q_s
            k_offset += kv_s
            v_offset += kv_s

        # golden data
        out = out.view(q_ntokens, head_num, head_size)
        return out.type(data_type)

    def undefined_golden_func(self, in_tensors):
        mixed_q, mixed_k, mixed_v = OpcheckUnpadSelfAttentionOperation.get_qkv(in_tensors)
        cache_k, cache_v, attention_mask, token_offset, seq_len, layerid = in_tensors[3], in_tensors[4], \
            in_tensors[5], in_tensors[6], in_tensors[7], int(in_tensors[8][0])
        self.bind_idx.extend([3, 4, 6, 7]) # cache_k, cache_v, token_offset, seq_len

        soc_version = self.get_soc_version()
        if soc_version != "Ascend910B":
            cache_k, cache_v = self.nz_2_nd(cache_k), self.nz_2_nd(cache_v)
            attention_mask = OpcheckUnpadSelfAttentionOperation.attention_mask_nz_2_nd(attention_mask, seq_len)
            self.in_tensors[3], self.in_tensors[4], self.in_tensors[5] = cache_k, cache_v, attention_mask

        batch_status = self.get_batch_status(in_tensors, seq_len)
        is_triu_mask = self.op_param.get("isTriuMask", 0)
        q_scale, qk_scale, head_info, _, _ = self.get_attention_params(mixed_q)
        head_num, head_size, _ = head_info

        offset = 0
        context_list = []
        for i in batch_status:
            cur_seqlen = seq_len[i]
            cur_token_offset = token_offset[i]
            cur_token_offset_start = cur_token_offset - cur_seqlen
            next_offset = offset + cur_seqlen
            cur_q, cur_k, cur_v = mixed_q[offset:next_offset], mixed_k[offset:next_offset], mixed_v[offset:next_offset]
            if cur_token_offset_start > 0:
                past_k = cache_k[layerid, i, :cur_token_offset_start, :]
                past_v = cache_v[layerid, i, :cur_token_offset_start, :]
                cur_k = torch.concat([past_k, cur_k], dim=0)
                cur_v = torch.concat([past_v, cur_v], dim=0)
            cur_q = (cur_q * q_scale).view(cur_seqlen, head_num, head_size).transpose(0, 1)
            cur_k = cur_k.view(cur_token_offset, head_num, head_size).permute(1, 2, 0)
            cur_qk = torch.bmm(cur_q, cur_k) # [head_num, seqlen, token_offset]
            clamp_type = self.op_param.get("clampType", ClampType.CLAMP_TYPE_UNDEFINED.value)
            if clamp_type == ClampType.CLAMP_TYPE_MIN_MAX.value:
                clamp_min, clamp_max = self.op_param.get("clampMin", 0.0), self.op_param.get("clampMax", 0.0)
                cur_qk = torch.clamp(cur_qk, clamp_min, clamp_max)
            if attention_mask.ndim == 3: # masked_fill
                cur_qk = cur_qk + attention_mask[i, :cur_seqlen, :cur_token_offset]
            else:
                cur_qk = cur_qk + attention_mask[:cur_seqlen, :cur_token_offset]
            cur_qk = cur_qk * qk_scale
            cur_qk = torch.nn.functional.softmax(cur_qk, dim=-1)
            cur_v = cur_v.view(cur_token_offset, head_num, head_size).transpose(0, 1)
            cur_context = torch.bmm(cur_qk, cur_v).transpose(0, 1).contiguous().view(cur_seqlen, head_num * head_size)
            context_list.append(cur_context)

            offset = next_offset

        out = torch.concat(context_list, dim=0)
        logger_text = f"context shape: {out.shape}"
        logger.debug(logger_text)
        return out

    def golden_calc(self, in_tensors):
        # KvCache配置，不支持calType为PA_ENCODER
        kvcache_cfg = self.op_param.get("kvcacheCfg", KvCacheCfg.K_CACHE_V_CACHE.value)
        logger_text = f"kvcacheCfg: {kvcache_cfg}"
        logger.debug(logger_text)

        # 直接传入kvcache
        if kvcache_cfg == KvCacheCfg.K_BYPASS_V_BYPASS.value:
            golden = self.kv_bypass_golden_func(in_tensors)
            return [golden]

        calc_type = self.op_param.get("calcType", CalcType.UNDEFINED.value)
        logger_text = f"CalcType: {calc_type}"
        logger.debug(logger_text)

        if calc_type == CalcType.PA_ENCODER.value:
            golden = self.pa_encoder_golden_func(in_tensors)
        else:
            golden = self.undefined_golden_func(in_tensors)
        return [golden]

    def test(self):
        self.execute()