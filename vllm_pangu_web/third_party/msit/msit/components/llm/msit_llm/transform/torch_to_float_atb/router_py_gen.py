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


def router_py_gen(parsed_model, save_name=None, save_dir=None):
    from msit_llm.transform.torch_to_float_atb import router_py_templates as templates

    weight_names = parsed_model.get('weight_names', {})
    model_name_lower = weight_names.get('model_name', '')
    model_name_capital = model_name_lower.capitalize()

    rr = ""
    rr += templates.COPYRIGHT_FORMATER.format(
        year=time.localtime().tm_year,
        licenses_url=get_public_url('msit_licenses_url')
    )

    check_if_safe_string(model_name_capital)
    check_if_safe_string(weight_names.get('pe_type'))
    rr += templates.IMPORT_FORMATER.format()
    rr += templates.CLASS_ROUTER_FORMATER.format(
        model_name_capital=model_name_capital,
        pe_type=weight_names.get('pe_type'),
        )

    save_name = utils.init_save_name(f"router_{model_name_lower}" if save_name is None else save_name) + ".py"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir=".")
    save_path = os.path.join(save_dir, save_name)
    write_file(save_path, rr)
    return save_path, rr
