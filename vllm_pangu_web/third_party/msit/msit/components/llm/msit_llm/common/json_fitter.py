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
import typing
import json
import base64

import pandas as pd
import onnx

from components.utils.check import Rule, validate_params
from components.utils.constants import TENSOR_MAX_SIZE
from components.utils.file_open_check import ms_open
from components.utils.file_utils import save_onnx
from msit_llm.common.log import logger
from msit_llm.common.utils import load_file_to_read_common_check


def atb_node_to_plain_node(atb_node_dict, level, target_level):
    if target_level != -1 and level >= target_level:
        return [atb_node_dict]

    # 递归元
    if "nodes" in atb_node_dict:
        plain_nodes = []
        for node_dict in atb_node_dict["nodes"]:
            plain_nodes = plain_nodes + atb_node_to_plain_node(node_dict, level + 1, target_level)
        return plain_nodes
    else:
        return [atb_node_dict]


def atb_json_dict_node_parse(atb_json_dict, target_level):
    plain_atb_nodes = []

    if target_level == 0 or "nodes" not in atb_json_dict:
        return [atb_json_dict]

    else:
        raw_atb_nodes = atb_json_dict["nodes"]
        level = 1
        for node in raw_atb_nodes:
            plain_atb_nodes = plain_atb_nodes + atb_node_to_plain_node(node, level, target_level)
        return plain_atb_nodes


def atb_param_to_onnx_attribute(atb_param_name, atb_param_value):
    onnx_attr_dict = {}
    onnx_attr_dict["name"] = atb_param_name

    if isinstance(atb_param_value, str):
        onnx_attr_dict["type"] = "STRINGS"
        try:
            onnx_attr_dict["strings"] = [base64.b64encode(atb_param_value.encode("utf-8")).decode("utf-8")]
        except UnicodeEncodeError:
            logger.debug("Unable to encode the base64 value of atb_param_values: %s", atb_param_value)
        except UnicodeDecodeError:
            logger.debug("Unable to decode the base64 value of atb_param_values: %s", atb_param_value)
        return onnx_attr_dict

    onnx_attr_dict["type"] = "FLOATS"
    values = []
    if isinstance(atb_param_value, list):
        for v in atb_param_value:
            values.append(float(v))
    elif atb_param_value:
        values.append(float(atb_param_value))
    onnx_attr_dict["floats"] = values
    return onnx_attr_dict


def parse_onnx_attr_from_atb_node_dict(atb_node_dict):
    onnx_attrs = []

    if not atb_node_dict.get("param", None):
        return onnx_attrs

    for param_name in atb_node_dict["param"]:
        if isinstance(atb_node_dict["param"][param_name], dict):
            for sub_param_name in atb_node_dict["param"][param_name]:
                full_name = param_name + "." + sub_param_name
                onnx_attr_dict = atb_param_to_onnx_attribute(full_name,
                                                             atb_node_dict["param"][param_name][sub_param_name])
                onnx_attrs.append(onnx_attr_dict)
        else:
            onnx_attr_dict = atb_param_to_onnx_attribute(param_name, atb_node_dict["param"][param_name])
        onnx_attrs.append(onnx_attr_dict)
    return onnx_attrs


def atb_node_to_onnx_node(atb_node_dict):
    onnx_node_dict = {}
    onnx_node_dict["name"] = atb_node_dict["opName"]
    onnx_node_dict["opType"] = atb_node_dict["opType"]
    onnx_node_dict["input"] = atb_node_dict["inTensors"]
    onnx_node_dict["output"] = atb_node_dict["outTensors"]
    onnx_node_dict["attribute"] = parse_onnx_attr_from_atb_node_dict(atb_node_dict)
    return onnx_node_dict


TYPESTR2ONNXTYPE = dict(
    float16=onnx.helper.TensorProto.FLOAT16,
    float32=onnx.helper.TensorProto.FLOAT,
    uint8=onnx.helper.TensorProto.UINT8,
    int8=onnx.helper.TensorProto.INT8,
    uint16=onnx.helper.TensorProto.UINT16,
    int16=onnx.helper.TensorProto.INT16,
    int32=onnx.helper.TensorProto.INT32,
    int64=onnx.helper.TensorProto.INT64,
    double=onnx.helper.TensorProto.DOUBLE,
    uint32=onnx.helper.TensorProto.UINT32,
    uint64=onnx.helper.TensorProto.UINT64,
    bfloat16=onnx.helper.TensorProto.BFLOAT16,
)


def build_onnx_shape_info(name, input_shape_info=None):
    if input_shape_info is None:
        return dict(name=name)
    else:
        input_type = input_shape_info.get("type")
        dims = input_shape_info.get("shape")
        return dict(
            name=name,
            type=dict(
                tensorType=dict(
                    elemType=TYPESTR2ONNXTYPE.get(input_type), shape=dict(dim=[dict(dimValue=d) for d in dims])
                )
            ),
        )


def atb_shape_to_onnx_shape(value_info, input_names, input_shapes):
    for input_index, input_shape_info in enumerate(input_shapes):
        if input_index >= len(input_names):
            break
        value_info.append(
            build_onnx_shape_info(input_names[input_index], input_shape_info)
        )


def atb_json_to_onnx_json(atb_json_dict, target_level, shape_contents):
    onnx_json_dict = {}
    plain_nodes = atb_json_dict_node_parse(atb_json_dict, target_level)

    for i, plain_node in enumerate(plain_nodes):
        plain_nodes[i] = atb_node_to_onnx_node(plain_node)

    onnx_json_dict["graph"] = {}
    onnx_json_dict["graph"]["node"] = plain_nodes

    if shape_contents is not None:
        onnx_json_dict["graph"]["valueInfo"] = []
        # csv_content like {nodename:inputs[{type, shape:[]}]}
        for node in plain_nodes:
            node_name = node.get("name")
            input_names = node.get("input")
            output_names = node.get("output")

            shape_info = shape_contents.get(node_name)
            if shape_info is None:
                continue

            atb_shape_to_onnx_shape(onnx_json_dict["graph"]["valueInfo"], input_names, shape_info.get("inputs", []))
            atb_shape_to_onnx_shape(onnx_json_dict["graph"]["valueInfo"], output_names, shape_info.get("outputs", []))

    value_info = {shape_info.get("name"): shape_info for shape_info in onnx_json_dict["graph"].get("valueInfo", [])}

    onnx_json_dict["graph"]["input"] = []
    for in_tensor_name in atb_json_dict["inTensors"]:
        onnx_input_tensor_dict = value_info.get(in_tensor_name, dict(name=in_tensor_name))
        onnx_json_dict["graph"]["input"].append(onnx_input_tensor_dict)

    onnx_json_dict["graph"]["output"] = []
    for out_tensor_name in atb_json_dict["outTensors"]:
        onnx_output_tensor_dict = value_info.get(out_tensor_name, dict(name=out_tensor_name))
        onnx_json_dict["graph"]["output"].append(onnx_output_tensor_dict)

    return onnx_json_dict


decorator_csv = validate_params(op_info_file=Rule.input_file())


@decorator_csv.to_return({}, logger)
def csv_to_content(op_info_file):
    op_info_file = load_file_to_read_common_check(op_info_file)
    if Rule.input_file().check(op_info_file, will_raise=True):
        pd_csv = pd.read_csv(op_info_file, sep="|")
    csv_content = {}  # csv_content like {nodename:inputs[{type, shape:[]}]}
    for index in range(len(pd_csv)):
        yy = pd_csv.iloc[index]
        node_name = yy.get("OpName")
        in_types = yy.get("InDType").split(";")
        in_shapes = [shape.split(",") for shape in yy.get("InShape").split(";")]
        out_types = yy.get("OutDType").split(";")
        out_shapes = [shape.split(",") for shape in yy.get("OutShape").split(";")]

        csv_content.setdefault(
            node_name,
            dict(
                inputs=[dict(type=t, shape=s) for t, s in zip(in_types, in_shapes)],
                outputs=[dict(type=t, shape=s) for t, s in zip(out_types, out_shapes)],
            ),
        )
    return csv_content


decorator_atb = validate_params(atb_json_path=Rule.input_file())


@decorator_atb.to_return(None, logger)
def atb_json_to_onnx(atb_json_path, target_level=-1, cache_csv_file: typing.Union[typing.Dict, None] = None):
    from google.protobuf.json_format import Parse

    atb_json_path = load_file_to_read_common_check(atb_json_path)
    with ms_open(atb_json_path, "r", max_size=TENSOR_MAX_SIZE) as file:
        json_content = json.loads(file.read(), parse_constant=lambda x: None)

    csv_content = None
    if cache_csv_file is not None:
        sub_pid = os.path.split(os.path.abspath(os.path.dirname(atb_json_path)))[-1]
        op_info_dir = os.path.join(os.path.dirname(atb_json_path), "..", "..", "operation_io_tensors", sub_pid)
        op_info_file = None
        if os.path.exists(op_info_dir):
            for file_name in os.listdir(op_info_dir):
                if file_name.startswith("operation") and file_name.endswith(".csv"):
                    op_info_file = os.path.join(op_info_dir, file_name)
                    break

        if op_info_file in cache_csv_file:
            csv_content = cache_csv_file.get(op_info_file)
        elif op_info_file is not None and os.path.exists(op_info_file):
            csv_content = csv_to_content(op_info_file)
            cache_csv_file.setdefault(op_info_file, csv_content)
        else:
            pass

    onnx_json = atb_json_to_onnx_json(json_content, target_level, csv_content)
    onnx_str = json.dumps(onnx_json)
    convert_model = Parse(onnx_str, onnx.ModelProto())
    onnx_dir = atb_json_path[0:-5] + ".onnx"
    save_onnx(convert_model, onnx_dir)
