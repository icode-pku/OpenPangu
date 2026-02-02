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


def float_model_cpp_gen(parsed_model, save_name=None, save_dir=None):
    from msit_llm.transform.torch_to_float_atb import float_model_cpp_templates as templates  # avoiding circular import

    model_name_lower = parsed_model.get("name", "model").lower()

    rr = ""
    rr += templates.COPYRIGHT_HEADER.format(
        year=time.localtime().tm_year,
        licenses_url=get_public_url('msit_licenses_url')
    )

    check_if_safe_string(model_name_lower)
    rr += templates.INCLUDE_HEADER_FORMATTER.format(
        model_name_lower=model_name_lower,
        other_operations="",
    )

    pre_properties = "\n".join([
        templates.WEIGHT_COUNT_FORMATTER.format(),
        templates.OPERATION_COUNT_FORMATTER.format(),
        templates.IN_TENSOR_ID_FORMATTER.format(),
        templates.INTERNAL_TENSOR_ID_FORMATTER.format(),
        templates.OUT_TENSOR_ID_FORMATTER.format(),
        templates.FROM_STRING_FORMATTER.format(),
    ])

    build_graph = templates.BUILD_GRAPH_FORMATTER.format(
        BUILD_GRAPH_PRE_PROCESS_FORMATTER=templates.BUILD_GRAPH_PRE_PROCESS_FORMATTER.format(),
        BUILD_GRAPH_PRE_PROCESS_NORM_FORMATTER=templates.BUILD_GRAPH_PRE_PROCESS_NORM_FORMATTER.format(),
        BUILD_GRAPH_LAYERS_FORMATTER=templates.BUILD_GRAPH_LAYERS_FORMATTER.format(model_name_lower=model_name_lower),
        BUILD_GRAPH_POST_PROCESS_NORM_FORMATTER=templates.BUILD_GRAPH_POST_PROCESS_NORM_FORMATTER.format(),
        BUILD_GRAPH_POST_PROCESS_LMHEAD_FORMATTER=templates.BUILD_GRAPH_POST_PROCESS_LMHEAD_FORMATTER.format(),
    )

    post_properties = "\n".join([
        templates.INFER_SHAPE_FORMATTER.format(),
        build_graph,
        templates.PARSE_PARAM_FORMATTER.format(),
        templates.BIND_PARAM_HOST_TENSOR_FORMATTER.format(),
    ])

    rr += templates.BASIC_CLASS_FORMATTER.format(
        model_name_lower=model_name_lower,
        pre_properties=pre_properties,
        post_properties=post_properties,
    )

    save_name = utils.init_save_name("decoder_model" if save_name is None else save_name) + ".cpp"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir="model")
    save_path = os.path.join(save_dir, save_name)
    write_file(save_path, rr)
    return save_path, rr
