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
import re


def is_word_embedding(name):
    return ('embed_tokens' in name or 'word_embeddings' in name) \
        and 'word_embeddings_layernorm' not in name


ATB_TORCH_BUILT_IN_OP_OUTPUT_MAPPING = {
    "LayerNormOperation": "LayerNorm",
    "LinearOperation": "Linear"
}


ATB_TORCH_CUSTOM_OP_OUTPUT_MAPPING = {
    "CommonLayer_outtensor0": ["GLMBlock_output_0", "BloomBlock_output_0"],
    "MlpGateLayerV2": ["BloomMLP", "MLP"],
    "RmsNormOperation": ["RMSNorm"],
    "SelfAttentionOperation": ["CoreAttention"],
}


ATB_QUANT_FLOAT_NODE_MAPPING = {
    "LinearQuantOperation": "LinearOperation",
    "LinearDequantOnly": "LinearNoQuant",
    "LinearRowParallelNoAdd": "LinearRowParallelAndAdd",
}


LAYER_OP_MAPPING_DICT = {
    'wordembedding': is_word_embedding,
    re.compile(r"^RmsNormOperation_\d+$"): 'root.model.norm',
    'lmhead': 'lm_head',
    'layernormoperation_35': 'embeddings_layernorm',
    'layernormoperation_66': 'ln_f'
}


QWEN_OP_MAPPING = {
        'qkv': ['self_attn.v_proj', 'self_attn.k_proj', 'self_attn.q_proj']
}


MODULE_MAPPING_DICT = {
    # ATB算子: [Pytorch算子列表]
    'Attention': ['self_attn', 'self_attention', 'Attention'],
    'Mlp': ['mlp', 'Mlp'],
    'mlp': ['mlp'],
    'self_attn': ['self_attn'],
    'self_attention': ['self_attention']
}