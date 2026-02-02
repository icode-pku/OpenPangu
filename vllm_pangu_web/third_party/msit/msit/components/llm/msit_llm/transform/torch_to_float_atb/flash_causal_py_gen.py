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


def match(keyword_lists, acl_inputs_name):
    for keyword_list in keyword_lists:
        for name in acl_inputs_name:
            if all(kw in name for kw in keyword_list):
                return name
    return 'PLACEHOLDER'


def flash_causal_py_gen(parsed_model, save_name=None, save_dir=None):
    from msit_llm.transform.torch_to_float_atb import flash_causal_py_templates as templates

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

    acl_inputs_name = parsed_model.get('acl_inputs_name', [])
    for name in acl_inputs_name:
        check_if_safe_string(name)

    acl_inputs_code_block = templates.ACL_INPUTS_CODE_BLOCK.format(
        acl_inputs_name=str(acl_inputs_name),
        input_ids=match(["INPUT ID".split(), "INPUT"], acl_inputs_name),
        input_embedding=match(["INPUT EMBEDDING".split()], acl_inputs_name),
        position_ids=match(["POSITION ID".split()], acl_inputs_name),
        cos_embed=match(["COS".split()], acl_inputs_name),
        sin_embed=match(["SIN".split()], acl_inputs_name),
        atten_mask=match(["ATTENTION MASK".split()], acl_inputs_name),
        block_tables=match(["BLOCK".split()], acl_inputs_name),
        slots=match(["SLOTS".split()], acl_inputs_name),
        input_lengths=match(["INPUT LENGTH".split(), "SEQ LEN".split()], acl_inputs_name),
        lm_head_indices=match(["LOGTIS INDICE".split(), "LOGITS INDICE".split()], acl_inputs_name),
        q_lens=match(["Q LEN".split()], acl_inputs_name),
    )

    check_if_safe_string(weight_names.get('model_prefix'))
    check_if_safe_string(weight_names.get('lmhead'))
    check_if_safe_string(weight_names.get('model_name_in_atb_framework').replace('/', '_'))
    check_if_safe_string(weight_names.get('qkv_sep'))
    check_if_safe_string(weight_names.get('mlp_sep'))
    rr += templates.CLASS_FLASH_CAUSAL_LM_FORMATER.format(
        model_name_capital=model_name_capital,
        model_prefix=weight_names.get('model_prefix'),
        lmhead=weight_names.get('lmhead'),
        model_name_in_atb_framework=weight_names.get('model_name_in_atb_framework').replace('/', '_'),
        qkv_sep=weight_names.get('qkv_sep'),
        mlp_sep=weight_names.get('mlp_sep'),
        acl_inputs_code_block=acl_inputs_code_block,
    )

    save_name = utils.init_save_name(f"flash_causal_{model_name_lower}" if save_name is None else save_name) + ".py"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir=".")
    save_path = os.path.join(save_dir, save_name)
    write_file(save_path, rr)
    return save_path, rr
