# Copyright (c) 2023-2025 Huawei Technologies Co., Ltd.
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

import torch

from components.utils.util import safe_get, safe_del
from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class CompressType(Enum):
    COMPRESS_TYPE_UNDEFINED = 0  # 默认值，不压缩
    COMPRESS_TYPE_KVHEAD = 1  # 压缩key_cache, value_cache的kvHead维度，只支持Atlas 800I A2


class MaskType(Enum):
    UNDEFINED = 0  # 默认值，全0的mask
    MASK_TYPE_NORM = 1  # 倒三角mask
    MASK_TYPE_ALIBI = 2  # alibi mask
    MASK_TYPE_SPEC = 3  # 并行解码mask


class QuantType(Enum):
    TYPE_QUANT_UNDEFINED = 0
    TYPE_DEQUANT_FUSION = 1


class CalcType(Enum):
    CALC_TYPE_UNDEFINED = 0  # 默认值，不开启并行解码
    CALC_TYPE_SPEC = 1  # 并行解码功能


class OpcheckPagedAttentionAttentionOperation(operation_test.OperationTest):
    @staticmethod
    def mask_nz_2_nd(mask, mask_type, context_lens, head_num):
        mask = mask.permute(0, 2, 1, 3)
        dim0, dim1, dim2, dim3 = mask.shape
        mask = mask.contiguous().view(dim0, dim1, dim2 * dim3)
        if mask_type == MaskType.MASK_TYPE_ALIBI.value:
            batch = len(context_lens)
            if dim0 != head_num:
                mask = mask.contiguous().view(batch, head_num, dim1, dim2 * dim3)
            else:
                mask = mask.contiguous().view(1, head_num, dim1, dim2 * dim3)
        return mask

    @staticmethod
    def get_fp32_b(i, b, fp32_input, is_k, has_bias):
        int8_b = b[i:, (i + 1), :, :, ]
        head_dim = int8_b.shape[2]
        int32_b = torch.matmul(torch.eye(int8_b.shape[1].type(torch.int32)), int8_b.type(torch.int32)).type(torch.int32)
        de_scale1_fp32, de_scale2_fp32, offset1, offset2 = fp32_input
        if is_k:
            if has_bias:
                int32_b = int32_b + offset1[i * head_dim: (i + 1) * head_dim]
            fp32_b = int32_b.type(torch.float32) * de_scale1_fp32[i * head_dim: (i + 1) * head_dim]
            fp32_b = torch.permute(fp32_b, (0, 2, 1))
        else:
            if has_bias:
                int32_b = int32_b + offset2[i * head_dim: (i + 1) * head_dim]
            fp32_b = int32_b.type(torch.float32) * de_scale2_fp32[i * head_dim: (i + 1) * head_dim]
        return fp32_b

    def get_quant_param(self, quant_type, has_quant_offset):
        if quant_type != QuantType.TYPE_QUANT_UNDEFINED.value:
            is_int8_flag = True
            if has_quant_offset:
                has_bias = True
                offset1 = self.in_tensors[8]
                de_scale1_fp32 = self.in_tensors[7]
                offset2 = self.in_tensors[10]
                de_scale2_fp32 = self.in_tensors[9]
            else:
                has_bias = False
                de_scale1_fp32 = self.in_tensors[7]
                de_scale2_fp32 = self.in_tensors[8]
                offset1 = None
                offset2 = None
        else:
            is_int8_flag = False
            has_bias = False
            de_scale1_fp32 = None
            de_scale2_fp32 = None
            offset1 = None
            offset2 = None
        return is_int8_flag, has_bias, [de_scale1_fp32, de_scale2_fp32, offset1, offset2]

    def group_matmul(self, head_num, kv_head_num, a, b, is_k):
        quant_type = self.op_param.get('quantType', QuantType.TYPE_QUANT_UNDEFINED.value)
        has_quant_offset = self.op_param.get('hasQuantOffset', False)
        is_int8_flag, has_bias, fp32_input = self.get_quant_param(quant_type, has_quant_offset)

        group_head = head_num // kv_head_num
        score = None
        for i in range(kv_head_num):
            fp32_a = a[i * group_head: (i + 1) * group_head, :, :].type(torch.float32)
            if is_int8_flag:
                fp32_b = OpcheckPagedAttentionAttentionOperation.get_fp32_b(i, b, fp32_input, is_k, has_bias)
                group_score = torch.matmul(fp32_a, fp32_b).type(torch.float16)
            else:
                group_score = torch.matmul(fp32_a, b[i: (i + 1), :, :].type(torch.float32))

            if score is None:
                score = group_score
            else:
                score = torch.concat((score, group_score), 0)
        logger_text = f"score shape: {score.shape}"
        logger.debug(logger_text)
        return score

    def ref_masked_attention(self, masked_attention_input, alibi_bias=None):
        query = masked_attention_input[0]  # (1, head_num, head_size)
        key = masked_attention_input[1]  # (context_len, kv_head_num, head_size)
        value = masked_attention_input[2]
        qk_scale = masked_attention_input[3]  # float

        # Q * K.T
        query = torch.permute(query, (1, 0, 2))
        quant_type = self.op_param.get('quantType', QuantType.TYPE_QUANT_UNDEFINED.value)
        if quant_type == QuantType.TYPE_QUANT_UNDEFINED.value:
            key = torch.permute(key, (1, 2, 0))  # 0 1 2
        else:
            key = torch.permute(key, (1, 0, 2))
        sim = self.group_matmul(query.shape[0], key.shape[0], query, key, 1)  # (head_num, q_seqlen, k_seqlen)
        if alibi_bias is None:
            sim = sim * qk_scale
        else:
            sim = sim * qk_scale
            sim = sim + alibi_bias
        # softmax
        row_max = torch.max(sim, axis=-1, keepdims=True).values
        sim -= row_max
        sim = torch.exp(sim)
        row_sum = torch.sum(sim, axis=-1, keepdims=True)
        p = sim / row_sum
        # P * V
        value = torch.permute(value, (1, 0, 2))
        out = self.group_matmul(query.shape[0], key.shape[0], p, value, 0)
        out = torch.permute(out, (1, 0, 2))
        return out

    def ref_single_query_cached_kv_attention(self, output, paged_input, mask) -> None:
        query, key_cache, value_cache, block_tables, context_lens = paged_input[:5]

        head_num = self.op_param.get('headNum', 0)
        kv_head_num = self.op_param.get('kvHeadNum', 0)
        num_tokens, _, head_size = query.shape
        _, block_size, _, _ = key_cache.shape
        max_context_len = max(context_lens)

        mask_type = self.op_param.get('maskType', MaskType.UNDEFINED.value)
        if mask_type != MaskType.UNDEFINED.value:
            mask_dim = 3 if mask_type == MaskType.MASK_TYPE_NORM.value else len(mask.shape)  # mask shape
        else:
            mask_dim = 0
        mask_index_coff = 1

        compress_type = self.op_param.get('compressType', CompressType.COMPRESS_TYPE_UNDEFINED.value)
        if compress_type == CompressType.COMPRESS_TYPE_KVHEAD.value:
            query = query.view(num_tokens * kv_head_num, head_num // kv_head_num, head_size)
            output = output.view(num_tokens * kv_head_num, head_num // kv_head_num, head_size)
            if mask_dim == 4:
                mask_shape = mask.shape
                mask = mask.view(mask_shape[0] * kv_head_num, head_num // kv_head_num, 1, max_context_len)
            else:
                mask_index_coff = kv_head_num

        for i in range(num_tokens):
            block_table, context_len = block_tables[i], int(context_lens[i])
            if context_len == 0:
                continue

            q, keys, values = query[i].view(1, head_num, head_size), [], []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
                k = key_cache[block_number, block_offset, :, :].reshape(kv_head_num, head_size)
                keys.append(k)
                v = value_cache[block_number, block_offset, :, :].reshape(kv_head_num, head_size)
                values.append(v)
            keys, values = torch.stack(keys, axis=0), torch.stack(values, axis=0)
            qk_scale = self.op_param.get('qkScale', 1.0)
            masked_attention_input = [q, keys, values, qk_scale]
            if mask_dim == 4:
                out = self.ref_masked_attention(masked_attention_input, mask[i, :, :1, :context_len])
                out = out.reshape(head_num, head_size)
            elif mask_dim == 3:
                out = self.ref_masked_attention(masked_attention_input, mask[i // mask_index_coff, :1, :context_len])
                out = out.reshape(head_num, head_size)
            else:
                out = self.ref_masked_attention(masked_attention_input, mask)
                out = out.reshape(head_num, head_size)
            output[i] = out

    def golden_calc(self, in_tensors):
        if len(in_tensors) < 5:
            raise ValueError(f"in_tensors length must be at least 5, but got {len(in_tensors)}")
        query, key_cache, value_cache, block_tables, context_lens = in_tensors[:5]
        self.bind_idx.append(4)
        mask = None

        head_num = self.op_param.get('headNum', 0)
        kv_head_num = self.op_param.get('kvHeadNum', 0)
        num_tokens, _, head_size = query.shape

        mask_type = self.op_param.get('maskType', MaskType.UNDEFINED.value)
        is_masked = mask_type != MaskType.UNDEFINED.value
        if is_masked:
            mask = safe_get(in_tensors, 5)
        if self.optimization_closed == "maskType":
            safe_del(self.in_tensors, 5)

        batch_run_status_enable = self.op_param.get("batchRunStatusEnable", False)
        if batch_run_status_enable:
            if is_masked:
                batch_status = safe_get(in_tensors, 6)
            else:
                batch_status = safe_get(in_tensors, 5)

        calc_type = self.op_param.get('calcType', CalcType.CALC_TYPE_UNDEFINED.value) # 暂时不使用
        if calc_type == CalcType.CALC_TYPE_SPEC.value:
            q_seq_lens = in_tensors[-1]
            self.bind_idx.append(len(in_tensors) - 1)

        soc_version = self.get_soc_version()
        if soc_version == 'Ascend310P':
            num_blocks, _, block_size, _ = key_cache.shape
            key_cache = key_cache.permute(0, 2, 1, 3).reshape(num_blocks, block_size, kv_head_num, head_size)
            value_cache = value_cache.permute(0, 2, 1, 3).reshape(num_blocks, block_size, kv_head_num, head_size)

            if mask_type != MaskType.UNDEFINED.value:
                mask = OpcheckPagedAttentionAttentionOperation.mask_nz_2_nd(mask, mask_type, context_lens, head_num)

        ref_output = torch.zeros_like(query)
        paged_input = [query, key_cache, value_cache, block_tables, context_lens]
        self.ref_single_query_cached_kv_attention(ref_output, paged_input, mask)
        return [ref_output]

    def test(self):
        self.execute()
