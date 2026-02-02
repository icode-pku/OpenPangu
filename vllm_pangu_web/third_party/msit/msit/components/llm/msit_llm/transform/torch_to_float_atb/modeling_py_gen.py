# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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

import os
import time

from components.utils.install import get_public_url
from msit_llm.transform.torch_to_float_atb import utils
from msit_llm.transform.utils import write_file, check_if_safe_string


def modeling_py_gen(parsed_model, save_name=None, save_dir=None):
    from msit_llm.transform.torch_to_float_atb import modeling_py_templates as templates

    weight_names = parsed_model.get('weight_names', {})
    model_name_lower = weight_names.get('model_name', '')
    model_name_capital = model_name_lower.capitalize()

    rr = ""
    rr += templates.COPYRIGHT_FORMATER.format(
        year=time.localtime().tm_year,
        licenses_url=get_public_url('msit_licenses_url')
    )

    check_if_safe_string(model_name_lower)
    check_if_safe_string(model_name_capital)
    rr += templates.IMPORT_FORMATER.format(
        model_name_lower=model_name_lower,
        model_name_capital=model_name_capital,
    )

    mlp_sep = weight_names.get('mlp_sep', [])
    for m in mlp_sep:
        check_if_safe_string(m)
    check_if_safe_string(weight_names.get('gate_up_proj'))
    check_if_safe_string(weight_names.get('post_attention_layernorm'))
    check_if_safe_string(weight_names.get('mlp_bias'))
    check_if_safe_string(weight_names.get('down_proj'))
    if len(mlp_sep) == 2:
        mlp_code_block = templates.MLP_SEP_FORMATER.format(
            gate_up_proj=weight_names.get('gate_up_proj'),
            gate_proj=mlp_sep[0],
            up_proj=mlp_sep[1],
            post_attention_layernorm=weight_names.get('post_attention_layernorm'),
            mlp_bias=weight_names.get('mlp_bias'),
            down_proj=weight_names.get('down_proj'),
        )
    else:
        check_if_safe_string(weight_names.get('layer_prefix'))
        mlp_code_block = templates.MLP_PACK_FORMATER.format(
            gate_up_proj=weight_names.get('gate_up_proj'),
            layer_prefix=weight_names.get('layer_prefix'),
            post_attention_layernorm=weight_names.get('post_attention_layernorm'),
            mlp_bias=weight_names.get('mlp_bias'),
            down_proj=weight_names.get('down_proj'),
        )

    qkv_sep = weight_names.get('qkv_sep', [])
    for q in qkv_sep:
        check_if_safe_string(q)
    check_if_safe_string(weight_names.get('query_key_value'))
    check_if_safe_string(weight_names.get('query_key_value_bias'))
    check_if_safe_string(weight_names.get('input_layernorm'))
    if len(qkv_sep) == 3:
        qkv_code_block = templates.QKV_SEP_FORMATER.format(
            q_proj=qkv_sep[0],
            k_proj=qkv_sep[1],
            v_proj=qkv_sep[2],
            query_key_value=weight_names.get('query_key_value'),
            query_key_value_bias=weight_names.get('query_key_value_bias'),
            input_layernorm=weight_names.get('input_layernorm'),
        )
    else:
        qkv_code_block = templates.QKV_PACK_FORMATER.format(
            query_key_value=weight_names.get('query_key_value'),
            query_key_value_bias=weight_names.get('query_key_value_bias'),
            input_layernorm=weight_names.get('input_layernorm'),
        )

    word_embeddings_layernorm = weight_names.get('word_embeddings_layernorm')
    if word_embeddings_layernorm is None:
        word_embeddings_layernorm_code_block = ''
    else:
        check_if_safe_string(word_embeddings_layernorm)
        word_embeddings_layernorm_code_block = templates.WORD_EMBEDDINGS_LATERNORM_FORMATER.format(
            model_name_capital=model_name_capital,
            word_embeddings_layernorm=word_embeddings_layernorm,
            RMSNormClass='RMSNormBias' if weight_names.get('layernorm_bias') else 'RMSNorm',        
            )

    check_if_safe_string(weight_names.get('model_prefix'))
    check_if_safe_string(weight_names.get('layers_prefix'))
    check_if_safe_string(weight_names.get('o_proj'))
    check_if_safe_string(weight_names.get('o_proj_bias'))
    check_if_safe_string(weight_names.get('mlp'))
    check_if_safe_string(weight_names.get('input_layernorm'))
    check_if_safe_string(weight_names.get('post_attention_layernorm'))
    check_if_safe_string(weight_names.get('layernorm'))
    check_if_safe_string(weight_names.get('word_embeddings'))
    check_if_safe_string(weight_names.get('self_attention'))
    rr += templates.CLASS_FLASH_MODEL_FORMATER.format(
        model_name_capital=model_name_capital,
        model_prefix=weight_names.get('model_prefix'),
        layers_prefix=weight_names.get('layers_prefix'),
        mlp_code_block=mlp_code_block,
        qkv_code_block=qkv_code_block,
        o_proj=weight_names.get('o_proj'),
        o_proj_bias=weight_names.get('o_proj_bias'),
        mlp=weight_names.get('mlp'),
        input_layernorm=weight_names.get('input_layernorm'),
        post_attention_layernorm=weight_names.get('post_attention_layernorm'),
        word_embeddings_layernorm_code_block=word_embeddings_layernorm_code_block,
        RMSNormClass='RMSNormBias' if weight_names.get('layernorm_bias') else 'RMSNorm',
        layernorm=weight_names.get('layernorm'),
        word_embeddings=weight_names.get('word_embeddings'),
        self_attention=weight_names.get('self_attention'),
    )

    save_name = utils.init_save_name(f"modeling_{model_name_lower}" if save_name is None else save_name) + ".py"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir=".")
    save_path = os.path.join(save_dir, save_name)
    write_file(save_path, rr)
    return save_path, rr
