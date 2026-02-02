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


def float_layer_h_gen(parsed_model, save_name=None, save_dir=None):
    from msit_llm.transform.torch_to_float_atb import float_layer_h_templates

    model_name_lower = parsed_model.get("name", "model").lower()

    rr = ""
    rr += float_layer_h_templates.COPYRIGHT_HEADER.format(
        year=time.localtime().tm_year,
        licenses_url=get_public_url('msit_licenses_url')
    )

    check_if_safe_string(model_name_lower)
    rr += float_layer_h_templates.INCLUDE_HEADER_FORMATTER.format(
        model_name_upper=model_name_lower.upper(),
    )

    rr += float_layer_h_templates.BASIC_CLASS_FORMATTER.format(
        model_name_lower=model_name_lower,
        STRUCT_PARAM_FORMATTER=float_layer_h_templates.STRUCT_PARAM_FORMATTER.format(),
        DECODER_LAYER_TENSOR_ID_FORMATTER=float_layer_h_templates.DECODER_LAYER_TENSOR_ID_FORMATTER.format()
    )

    save_name = utils.init_save_name(save_name if save_name else "decoder_layer") + ".h"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir="layer")
    save_path = os.path.join(save_dir, save_name)
    write_file(save_path, rr)
    return save_path, rr
