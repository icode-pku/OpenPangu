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


import string
from collections import namedtuple

import torch



CONFIG_ATTR_CANDIDATES = {
    "num_hidden_layers": ["num_hidden_layers", "num_layers", "n_layers"],
    "num_attention_heads": ["num_attention_heads"],
    "num_key_value_heads": ["num_key_value_heads"],
    "hidden_size": ["hidden_size"],
    "rms_norm_eps": ["rms_norm_eps"],
    "rope_theta": ["rope_theta"],
    "vocab_size": ["vocab_size"],
    "text_config": ["text_config", "llm_config"],
}

NN_MODULE_STACK = "nn_module_stack"
SKIP_NODES = ["size", "getitem", "to", "float", "finfo", "dropout"]
SKIP_MODULES = ["Dropout"]
TORCH_MODULE_TO_ATB_MAP = {
    "Embedding": dict(op_type="Gather", op_param={}, is_weights_first=True),
    "Gather": dict(op_type="Gather", op_param={}),
    ".*RMSNorm$": dict(op_type="RmsNorm", op_param={"layerType": "RMS_NORM_NORM", "epsilon": 1e-5}),
    ".{0,100}LayerNorm$": dict(
        op_type="LayerNorm",
        op_param={"layerType": "LAYER_NORM_NORM", "normParam": {"beginParamsAxis": 1, "beginNormAxis": 1}},
    ),
    "Linear": dict(op_type="Linear", op_param={"hasBias": False, "enAccum": False}),
    ".*Rotary.*": dict(op_type="Rope", op_param={"rotaryCoeff": 2}),
    ".*Attention$": dict(
        op_type="SelfAttention",
        op_param={"headNum": 1, "kvHeadNum": 1, "calcType": "PA_ENCODER", "qkScale": 1, "maskType": "MASK_TYPE_NORM"},
    ),
    "SiLU.{0,100}": dict(op_type="Activation", op_param={"activationType": "ACTIVATION_SWISH"}),
    "Gelu.{0,100}": dict(op_type="Activation", op_param={"activationType": "ACTIVATION_GELU"}),
    ".{0,100}Gelu": dict(op_type="Activation", op_param={"activationType": "ACTIVATION_GELU"}),
    "add": dict(op_type="Elewise", op_param={"elewiseType": "ELEWISE_ADD"}),
    "mul": dict(op_type="Elewise", op_param={"elewiseType": "ELEWISE_MUL"}),
}

_FX_OP_TYPES = ["call_method", "call_module", "call_function", "placeholder", "output", "get_attr"]
FX_OP_TYPES = namedtuple("FX_OP_TYPES", _FX_OP_TYPES)(*_FX_OP_TYPES)

_FIXED_INPUTS = {
    "input_ids",
    "position_ids",
    "inputs_embeds",
    "cos_table",
    "sin_table",
    "slots_mapping",
    "attention_mask",
    "seq_len",
}
FIXED_INPUTS = namedtuple("FIXED_INPUTS", _FIXED_INPUTS)(*_FIXED_INPUTS)
KV_CACHE_SURFFIX = namedtuple("FIXED_INPUTS", ["k_cache", "v_cache"])("k_cache", "v_cache")
BASIC_INPUT_NAMES = (FIXED_INPUTS.input_ids,)

_RESHPAE_KIND = ["reshape_qkv", "reshape_0_12"]
RESHPAE_KIND = namedtuple("RESHPAE_KIND", _RESHPAE_KIND)(*_RESHPAE_KIND)

VALID_NAME_CHARS = string.ascii_letters + string.digits + "_."
FLOAT_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

MINDIE_ATB_MODEL = {
    "mixtral8": "mixtral"
}
GATE_UP_WEIGHT = "gate_up_weight_"
DOWN_WEIGHT = "down_weight_"

