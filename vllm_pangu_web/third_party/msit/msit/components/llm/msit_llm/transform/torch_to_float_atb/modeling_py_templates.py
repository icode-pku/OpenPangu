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
import os
import math
from typing import Optional, List, Tuple

import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    PositionRotaryEmbedding,
    TensorEmbedding,
    KvCache,
    TensorParallelEmbedding,
    load_column_multi,
    paged_attn,
    flash_attn,
    reshape_and_cache
)
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type, QuantType

EMBEDDING_PARALLEL_THRESHOLD = 128256 # vocab size of {model_name_capital}
"""

MLP_PACK_FORMATER = """        
        linear_names = [f'{{prefix}}.{gate_up_proj}']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{{layer_prefix}}.{post_attention_layernorm}'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        # Fuse gate and up proj
        if config.quantize == QuantType.W8A8SC:
            self.gate_up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{{prefix}}.{gate_up_proj}",
                weights=weights,
                bias={mlp_bias},
            )
        else:
            self.gate_up_proj = TensorParallelColumnLinear.load_gate_up(
                config,
                prefix=f"{{prefix}}.{gate_up_proj}",
                weights=weights,
                bias={mlp_bias},
            )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{{prefix}}.{down_proj}",
            weights=weights,
            bias={mlp_bias},
        )
"""

MLP_SEP_FORMATER = """
        linear_names = [f'{{prefix}}.{up_proj}', f'{{prefix}}.{gate_proj}']
        pack_name = f'{{prefix}}.{gate_up_proj}'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{{layer_prefix}}.{post_attention_layernorm}'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI
        ]:
            self.gate_up_proj = load_column_multi(
                config,
                prefixes=[f"{{prefix}}.{gate_proj}", f"{{prefix}}.{up_proj}"],
                weights=weights,
                head_size=1,
                bias={mlp_bias},
            )
        elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
            self.gate_up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{{prefix}}.{gate_up_proj}",
                weights=weights,
                bias={mlp_bias},
            )
        else:
            self.gate_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{{prefix}}.{gate_proj}",
                weights=weights,
                bias={mlp_bias},
            )
            self.up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{{prefix}}.{up_proj}",
                weights=weights,
                bias={mlp_bias},
            )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{{prefix}}.{down_proj}",
            weights=weights,
            bias={mlp_bias},
        )
"""

QKV_PACK_FORMATER = """ 
        linear_names = [f'{{prefix}}.{query_key_value}']
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{{layer_prefix}}.{input_layernorm}'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        if config.quantize == QuantType.W8A8SC:
            self.query_key_value = TensorParallelColumnLinear.load(
                config,
                prefix=f"{{prefix}}.{query_key_value}",
                weights=weights,
                bias={query_key_value_bias}
            )
        else:
            self.query_key_value = TensorParallelColumnLinear.load_qkv(
                config,
                prefix=f"{{prefix}}.{query_key_value}",
                weights=weights,
                bias={query_key_value_bias},
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
            )
"""

QKV_SEP_FORMATER = """
        if config.kv_quant is not None:
            self.k_cache_quant = KvCache.load(prefix=f"{{prefix}}.{k_proj}", weights=weights)
            self.v_cache_quant = KvCache.load(prefix=f"{{prefix}}.{v_proj}", weights=weights)

        linear_names = [f'{{prefix}}.{q_proj}', f'{{prefix}}.{k_proj}', f'{{prefix}}.{v_proj}']
        pack_name = f'{{prefix}}.{query_key_value}'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{{layer_prefix}}.{input_layernorm}'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI
        ]:            
            self.query_key_value = load_column_multi(
                config,
                prefixes=[f"{{prefix}}.{q_proj}", 
                          f"{{prefix}}.{k_proj}", 
                          f"{{prefix}}.{v_proj}"],
                weights=weights,
                head_size=self.head_size,
                bias={query_key_value_bias}
            )
        elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
            self.query_key_value = TensorParallelColumnLinear.load(
                config,
                prefix=f"{{prefix}}.{query_key_value}",
                weights=weights,
                bias={query_key_value_bias},
            )
        else:
            self.q_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{{prefix}}.{q_proj}",
                weights=weights,
                bias={query_key_value_bias},
            )
            self.k_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{{prefix}}.{k_proj}",
                weights=weights,
                bias={query_key_value_bias},
            )
            self.v_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{{prefix}}.{v_proj}",
                weights=weights,
                bias={query_key_value_bias},
            )
"""

WORD_EMBEDDINGS_LATERNORM_FORMATER = """
        self.word_embeddings_layernorm = {model_name_capital}{RMSNormClass}(
            prefix=f"{word_embeddings_layernorm}", weights=weights
        )
"""

CLASS_FLASH_MODEL_FORMATER = """
class {model_name_capital}RMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        weight = weights.get_tensor(f"{{prefix}}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 8192:
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states, residual


class {model_name_capital}RMSNormBias(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        weight = weights.get_tensor(f"{{prefix}}.weight")
        try:
            bias = weights.get_tensor(f"{{prefix}}.bias")
        except AssertionError:
            bias = torch.zeros(weight.shape, dtype=weights.dtype)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.variance_epsilon = eps


class {model_name_capital}RMSNormWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        self.ori = {model_name_capital}RMSNorm(prefix, weights, eps)
        self.anti = {model_name_capital}RMSNormBias(f'{{prefix}}.module', weights, eps)


class {model_name_capital}RMSNormAntiOutlierWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        self.ori = {model_name_capital}RMSNorm(f'{{prefix}}.ori', weights, eps)
        self.anti = {model_name_capital}RMSNormBias(f'{{prefix}}.anti', weights, eps)


class {model_name_capital}MLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        {mlp_code_block}


class Flash{model_name_capital}Attention(torch.nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0, device="cpu").to(
            weights.device)
        self.softmax_scale = self.head_size ** -0.5

        # can not support self.num_heads % weights.process_group.size() != 0
        if (config.num_attention_heads != config.num_key_value_heads
                and (self.num_heads % weights.process_group.size() != 0)):
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {{self.num_heads}} "
                f"and `num_shards`: {{weights.process_group.size()}}"
            )
        if config.num_key_value_heads < weights.process_group.size():
            repeat_times = weights.process_group.size() // config.num_key_value_heads
        else:
            repeat_times = 1

        self.num_heads = (self.num_heads + weights.process_group.size() - 1) // weights.process_group.size()
        if config.num_key_value_heads != config.num_attention_heads:
            self.num_key_value_heads = config.num_key_value_heads * repeat_times
            self.num_key_value_heads = self.num_key_value_heads // weights.process_group.size()
        else:
            self.num_key_value_heads = self.num_heads

        {qkv_code_block}
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{{prefix}}.{o_proj}",
            weights=weights,
            bias={o_proj_bias},
            gqa_size=self.head_size,
        )
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

        self.prefix = prefix


class Flash{model_name_capital}Layer(nn.Module):
    def __init__(self, layer_id, config, weights, model_prefix="model"):
        super().__init__()
        prefix = f"{layers_prefix}.{{layer_id}}"
        self.self_attention = Flash{model_name_capital}Attention(
            prefix=f"{{prefix}}.{self_attention}", config=config, weights=weights
        )
        self.mlp = {model_name_capital}MLP(prefix=f"{{prefix}}.{mlp}", config=config, weights=weights)
        if self.self_attention.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16]:
            self.input_layernorm = {model_name_capital}{RMSNormClass}(
                prefix=f"{{prefix}}.{input_layernorm}", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attention.pack_type in [
            PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
            PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
            PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI
        ]:
            self.input_layernorm = {model_name_capital}RMSNormWrapper(
                prefix=f"{{prefix}}.{input_layernorm}", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attention.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            self.input_layernorm = {model_name_capital}RMSNormAntiOutlierWrapper(
                prefix=f"{{prefix}}.{input_layernorm}", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attention.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                          PackType.MIX_W8A8SC]:
            self.input_layernorm = {model_name_capital}RMSNormBias(
                prefix=f"{{prefix}}.{input_layernorm}", weights=weights, eps=config.rms_norm_eps
            )
        else:
            raise AssertionError(f'self_attention.pack_type: {{self.self_attention.pack_type}} not supported')
        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16]:
            self.post_attention_layernorm = {model_name_capital}{RMSNormClass}(
                prefix=f"{{prefix}}.{post_attention_layernorm}",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        elif self.mlp.pack_type in [
            PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
            PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
            PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI
        ]:
            self.post_attention_layernorm = {model_name_capital}RMSNormWrapper(
                prefix=f"{{prefix}}.{post_attention_layernorm}",
                weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            self.post_attention_layernorm = {model_name_capital}RMSNormAntiOutlierWrapper(
                prefix=f"{{prefix}}.{post_attention_layernorm}",
                weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                    PackType.MIX_W8A8SC]:
            self.post_attention_layernorm = {model_name_capital}RMSNormBias(
                prefix=f"{{prefix}}.{post_attention_layernorm}",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        else:
            raise AssertionError(f'mlp.pack_type: {{self.mlp.pack_type}} not supported')


class Flash{model_name_capital}Model(torch.nn.Module):
    def __init__(self, config, weights, model_prefix="model"):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.parallel_embedding = config.vocab_size >= EMBEDDING_PARALLEL_THRESHOLD
        self.word_embeddings = (TensorParallelEmbedding if self.parallel_embedding else TensorEmbedding)(
            prefix=f"{word_embeddings}", weights=weights
        )
        {word_embeddings_layernorm_code_block}
        self.layers = nn.ModuleList(
            [
                Flash{model_name_capital}Layer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.layernorm = {model_name_capital}{RMSNormClass}(
            prefix=f"{layernorm}", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attention.head_size
        self.num_heads = self.layers[0].self_attention.num_heads
        self.num_key_value_heads = self.layers[0].self_attention.num_key_value_heads

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
"""