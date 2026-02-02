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

COPYRIGHT_FORMATER = """# Copyright {year} EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     {licenses_url}
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

IMPORT_FORMATER = """
import json
import math
import os
from typing import Optional, List, Tuple

import torch
import torch_npu

from atb_llm.utils.log import logger
from .modeling_{model_name_lower} import Flash{model_name_capital}Model
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.layers import load_column_multi, TensorHead
from atb_llm.utils.dist import get_rank_table_file, get_backend
"""


ACL_INPUTS_CODE_BLOCK = """
        acl_inputs = {{
            "input_ids": self.placeholder if self.skip_word_embedding else input_ids,
            "input_embedding": input_ids if self.skip_word_embedding else self.placeholder,
            "position_ids": position_ids.to(torch.int64),
            "cos_embed": self.cos_embed,
            "sin_embed": self.sin_embed,
            "atten_mask": atten_mask,
            "block_tables": block_tables.to(torch.int32),
            "slots": slots.to(torch.int32),
            "input_lengths": input_lengths.to(torch.int32),
        }}

        if is_prefill:
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                               dtype=torch.int64, device=input_ids.device)
            acl_inputs["lm_head_indices"] = lm_head_indices.to(torch.int64)
            acl_param = json.dumps({{
                "seqLen": input_lengths.tolist()
            }})
        else:
            acl_inputs["input_ids"] = input_ids
            acl_inputs["input_embedding"] = self.placeholder
            acl_inputs["lm_head_indices"] = self.lm_head_indices_fake
            
            q_lens = kwargs.get('q_lens', [])
            if self.speculate_enable:
                acl_inputs["q_lens"] = torch.tensor(q_lens).to(self.device).to(torch.int32)

            acl_param = json.dumps({{
                "seqLen": input_lengths.tolist(),
                "qLen": q_lens
            }})
        
        # Please check if acl_inputs_name is correct.
        acl_inputs_name = {acl_inputs_name}

        acl_inputs_name_list = match_table(acl_inputs_name)
        acl_inputs_list = [acl_inputs.get(name, self.placeholder) for name in acl_inputs_name_list]
"""

CLASS_FLASH_CAUSAL_LM_FORMATER = """
def match(dic, name):
    for cand, keyword_lists in dic.items():
        for keyword_list in keyword_lists:
            if all(kw in name for kw in keyword_list):
                return cand
    return ''

    
def match_table(acl_inputs_name):
    name_table = {{
        'input_embedding':["INPUT EMBEDDING".split()],
        'input_ids':["INPUT ID".split(), "INPUT"],
        'position_ids':["POSITION ID".split()],
        'cos_embed':["COS".split()],
        'sin_embed':["SIN".split()],
        'atten_mask':["ATTENTION MASK".split()],
        'block_tables':["BLOCK".split()],
        'slots':["SLOTS".split()],
        'input_lengths':["INPUT LENGTH".split(), "SEQ LEN".split()],
        'lm_head_indices':["LOGTIS INDICE".split(), "LOGITS INDICE".split()],
        'q_lens':["Q LEN".split()],
    }}
    res = []
    
    for name in acl_inputs_name:
        name = name.upper()
        match_result = match(name_table, name)
        res.append(match_result)
    return res


class Flash{model_name_capital}ForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights):
        super().__init__(config, weights)

        self.model = Flash{model_name_capital}Model(config, weights, "{model_prefix}")
        if self.quantize == "w8a8sc":
            self.lm_head = TensorHead.load_weight(
                config,
                prefix="{lmhead}",
                weights=weights,
                is_norm=False,
            )
        else:
            self.lm_head = load_column_multi(
                config,
                prefixes=["{lmhead}"],
                weights=weights,
                head_size=1,
                lm_head=True,
            )

        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.total_head_nums = config.hidden_size // self.head_dim

        self.placeholder = torch.tensor([0], dtype=self.dtype, device="npu")
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({{}})
        self.transdata_operation.set_param(self.transdata_param)
        self.position_embedding_type = config.pe_type
        self.alibi_bias_max = config.alibi_bias_max
        self.rope_keep_local_base_windows = config.rope_keep_local_base_windows
        self.rope_vanilla_theta = config.rope_vanilla_theta
        self.rope_mscale = config.rope_mscale
        self.rope_given_inv_feq_str = config.rope_given_inv_feq_str
        self.atten_mask_cpu = None
        self.alibi_mask_compress = False
        self.skip_word_embedding = False
        if self.position_embedding_type != "ROPE" and self.position_embedding_type != "ALIBI":
            logger.error("error: only support petype: ROPE and ALIBI, check your config.json: pe_type")
            raise AssertionError(f'petype: {{self.position_embedding_type}} not supported')
        self.cos_embed = None
        self.sin_embed = None
        self.wins_batch_1 = None
        self.decoder_slots = None
        self.all_wins_batch = None
        self.block_tables_global = None
        self.wins_global = None

        if 'ChatGLMModel' in self.config.architectures:
            class RotaryEmbedding(torch.nn.Module):
                def __init__(self, dim, rope_ratio=1, original_impl=False, device=None, dtype=None, version=None):
                    super().__init__()
                    inv_freq = 1.0 / \
                            (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
                    self.register_buffer("inv_freq", inv_freq)
                    self.dim = dim
                    self.original_impl = original_impl
                    self.rope_ratio = rope_ratio
                    self.version = version

                def forward_impl(
                        self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
                ):
                    theta = 1.0 / \
                            (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

                    # Create position indexes `[0, 1, ..., seq_len - 1]`
                    seq_idx = torch.arange(seq_len, dtype=dtype, device=device) / self.rope_ratio
                    if self.version == 'v3_6b':
                        base = base * self.rope_ratio
                        theta = 1.0 / \
                                (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

                        # Create position indexes `[0, 1, ..., seq_len - 1]`
                        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

                    # Calculate the product of position index and $\theta_i$
                    idx_theta = torch.outer(seq_idx, theta).float()

                    emb = torch.stack((idx_theta, idx_theta), dim=-1)
                    rope_cos = torch.cos(emb)
                    rope_sin = torch.sin(emb)

                    # this is to mimic the behaviour of complex32, else we will get different results
                    if dtype in (torch.float16, torch.bfloat16, torch.int8):
                        if dtype == torch.bfloat16:
                            rope_cos = rope_cos.bfloat16()
                            rope_sin = rope_sin.bfloat16()
                        else:
                            rope_cos = rope_cos.half()
                            rope_sin = rope_sin.half()

                    return rope_cos, rope_sin

                def forward(self, max_seq_len):
                    return self.forward_impl(
                        max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
                    )
            
            rotary_dim = self.head_dim
            self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, rope_ratio=self.config.rope_ratio, device=weights.device,
                                        dtype=config.torch_dtype)

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config):
        # 初始化模型
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("{model_name_in_atb_framework}")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("{model_name_in_atb_framework}")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attention',
            pack_name='query_key_value',
            sep_names={qkv_sep},
            o_name='o_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names={mlp_sep},
            down_name='down_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.model.word_embeddings)
        if hasattr(self.model, 'word_embeddings_layernorm'):
            weight_wrapper.register_model_norm(self.model.word_embeddings_layernorm)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.self_attention
                del layer.post_attention_layernorm
                del layer.mlp
            if self.kv_quant is not None:
                weight_wrapper.register_layer_kvquant(layer)
        weight_wrapper.register_model_norm(self.model.layernorm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        # 设置模型参数
        rank_table_file = get_rank_table_file()
        backend = get_backend(self.soc_info.need_nz)
        coder_param = {{
            "rmsNormEps": self.config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "skipWordEmbedding": False,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": self.model.parallel_embedding,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "supportSwiGLU": False if self.soc_info.need_nz else True,
            "kvQuant": self.kv_quant is not None,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": backend,
            "rankTableFile": rank_table_file,
            "positionEmbeddingType": self.position_embedding_type,
            "hiddenSize": self.hidden_size,
            "gemma": False,
            "supportCompressHead": self.compress_head_enable,

            "layerNormEps": self.config.rms_norm_eps,
            "weightQuantType": 'float',
        }}
        encoder_param = {{
            **coder_param, "isPrefill": True,
            "supportLcoc": self.lcoc_enable,
            "supportSpeculate": False,
            "skipWordEmbedding": self.skip_word_embedding
        }}
        decoder_param = {{
            **coder_param, "isPrefill": False, "supportLcoc": False,
            "supportSpeculate": self.speculate_enable
        }}
        self.acl_encoder_operation.set_param(json.dumps({{**encoder_param}}))
        self.acl_decoder_operation.set_param(json.dumps({{**decoder_param}}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def init_cos_sin_table(self, max_seq_len, dim, dtype, device):    
        if 'ChatGLMModel' in self.config.architectures:
            self.cos_embed, self.sin_embed = self.rotary_pos_emb.forward(self.config.max_position_embeddings)
            return
        if self.rope_given_inv_feq_str is None and self.rope_vanilla_theta is None:
            self._init_rope_cos_sin(max_seq_len, dtype, device)
        else:
            self.cos_embed, self.sin_embed = self._get_cos_sin_table(
                max_seq_len, dim, dtype, device, 0, self.rope_mscale,
                self.rope_keep_local_base_windows, self.rope_theta,
                self.rope_vanilla_theta, self.rope_given_inv_feq_str
            )

    def get_atten_mask(self, max_seq_len, position_ids, is_prefill, **kwargs):
        n_head = self.num_attention_heads
        head_nums = self.config.num_attention_heads // self.tp_world_size
        
        if is_prefill:
            atten_mask = None
            if self.position_embedding_type == "ROPE":
                self.init_cos_sin_table(self.max_position_embeddings, self.head_dim, self.dtype, self.device)

                atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, self.dtype,
                                                          self.device)
                if self.soc_info.need_nz:
                    atten_mask = self.transdata_operation.execute([atten_mask])[0]                    
            elif self.position_embedding_type == "ALIBI":
                self.cos_embed = self.placeholder
                self.sin_embed = self.placeholder
                if self.atten_mask_cpu is None:
                    self.atten_mask_cpu = self._gen_alibi_mask(self.total_head_nums, self.max_position_embeddings,
                                                               self.alibi_bias_max)[
                                          self.tp_rank * n_head:(self.tp_rank + 1) * n_head, :, :].to(self.dtype)
                if self.alibi_mask_compress:
                    # 算子要求: 小于128则按实际长度切，大于128则按128切，算子内部扩展到实际长度
                    slice_len = max_seq_len if max_seq_len <= 128 else 128
                    atten_mask = self.atten_mask_cpu[:, :, :slice_len].npu()
                else:
                    atten_mask = self.atten_mask_cpu[:, :max_seq_len, :max_seq_len].npu()
            else:
                logger.error(f"position_embedding_type is inllegal {{self.position_embedding_type}}")

        else:
            q_lens = kwargs.get('q_lens', [])
            spec_mask = kwargs.get('spec_mask', None)
            atten_mask = None
            if self.speculate_enable and self.soc_info.need_nz:
                spec_mask = self.transdata_operation.execute([spec_mask])[0]

            if self.position_embedding_type == "ROPE":
                atten_mask = spec_mask if self.speculate_enable else self.attn_mask_fake
            elif self.position_embedding_type == "ALIBI":
                atten_mask = self._gen_alibi_mask_decoder(self.total_head_nums, position_ids.tolist(),
                                        max_seq_len, self.alibi_bias_max)[:,
                                        self.tp_rank * n_head:(self.tp_rank + 1) * n_head, :, :].to(self.dtype).npu()
            else:
                logger.error(f"position_embedding_type is inllegal {{self.position_embedding_type}}") 
        return atten_mask

    def prepare_inputs_for_ascend(
            self, input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
            **kwargs
    ):
        atten_mask = self.get_atten_mask(max_seq_len, position_ids, is_prefill, **kwargs)
        {acl_inputs_code_block}
        return acl_inputs_list, acl_param

    def _get_interleave(self, n, alibi_bias_max=8.0):
        def _get_interleave_power_of_2(n, alibi_bias_max):
            if n == 0:
                return 0
            start = (0.5 ** (alibi_bias_max / n))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return _get_interleave_power_of_2(n, alibi_bias_max)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return _get_interleave_power_of_2(closest_power_of_2, alibi_bias_max) + \
                self._get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    def _fill_with_neg_inf(self, t):
        return t.float().fill_(float("-inf")).type_as(t)

    def _gen_alibi_mask(self, n_head, max_pos, alibi_bias_max=8.0):
        slopes = torch.Tensor(self._get_interleave(n_head, alibi_bias_max))
        tensor_list = []
        # 算子要求的压缩alibi mask shape为 [head_num, max_seq, 128]
        for i in range(128):
            tensor = torch.empty(max_pos).fill_(-float('inf'))
            tensor[i:] = -1 * torch.arange(0, max_pos - i)
            tensor = tensor.unsqueeze(0)
            tensor_list.append(tensor)
        tensor = torch.cat(tensor_list, dim=0).t()
        tensor = tensor.expand(n_head, -1, -1)
        alibi_mask = slopes.unsqueeze(1).unsqueeze(1) * tensor
        return alibi_mask

    def _gen_alibi_mask_decoder(self, n_head, pos_list, max_pos, alibi_bias_max=8.0):
        slopes = torch.Tensor(self._get_interleave(n_head, alibi_bias_max))
        tensor_list = []
        for pos in pos_list:
            tensor = torch.empty(max_pos).fill_(-float('inf'))
            tensor[:pos + 1] = torch.arange(-pos, 1)
            tensor = tensor.unsqueeze(0)
            tensor_list.append(tensor)
        tensor = torch.cat(tensor_list, dim=0)
        tensor = tensor.expand(n_head, -1, -1)
        alibi_mask = slopes.unsqueeze(1).unsqueeze(1) * tensor
        return alibi_mask.permute(1, 0, 2).unsqueeze(2)

    # 固定基频: rope_theta
    # 自定义基频: rope_given_inv_feq_str
    # 分段基频: rope_theta/rope_given_inv_feq_str + rope_vanilla_theta + rope_keep_local_base_windows
    def _get_cos_sin_table(self, max_seq_len, dim, dtype, device, offset=0, mscale=1,
                           keep_local_base_windows=None, rope_theta=None, rope_vanilla_theta=None,
                           given_inv_feq_str=None):

        if given_inv_feq_str:
            inv_freq = torch.FloatTensor([float(invf) for invf in given_inv_feq_str.split(',')], device=device)
            if len(inv_freq) != dim // 2:
                raise AssertionError(f'given_inv_feq_str: length not match head_dim/2')
        else:
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

        seq = torch.arange(max_seq_len, device=device).float() + offset
        freqs = torch.outer(seq, inv_freq)

        if keep_local_base_windows:
            keep_local_base_windows = [int(w) for w in keep_local_base_windows.split(',')]
            if len(keep_local_base_windows) != dim // 2:
                raise AssertionError(f'keep_local_base_windows: length not match head_dim/2')

            inv_freq_base = 1.0 / (rope_vanilla_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
            freqs_base = torch.outer(seq, inv_freq_base)
            freqs_after_window = freqs + torch.tensor(keep_local_base_windows) * (inv_freq_base - inv_freq)
            for idx, i_keep_local_base_window in enumerate(keep_local_base_windows):
                freqs[:, idx] = torch.cat((
                    freqs_base[:i_keep_local_base_window, idx],
                    freqs_after_window[i_keep_local_base_window:, idx]
                ))

        # Different from paper, but it uses a different permutation in order to obtain the same calculation（ks）
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * mscale).to(dtype).to(device), (emb.sin() * mscale).to(dtype).to(device)

    def _init_rope_cos_sin(self, max_seq_len, dtype, device):
        if not hasattr(self.config, 'rope_scaling') or self.config.rope_scaling is None:
            self.rotary_embedding.update_cos_sin_cache_total(dtype,
                                                             device,
                                                             max_seq_len)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_embedding.update_cos_sin_cache_total(dtype,
                                                                device,
                                                                max_seq_len)
            elif scaling_type == "dynamic":
                raise ValueError(f"not support RoPE scaling type {{scaling_type}}")
            else:
                raise ValueError(f"Unknown RoPE scaling type {{scaling_type}}")

        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()
"""