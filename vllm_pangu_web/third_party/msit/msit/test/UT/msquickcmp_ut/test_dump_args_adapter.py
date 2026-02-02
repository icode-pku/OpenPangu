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
# Wcmp_process.pyITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import os
from unittest.mock import patch, MagicMock
from msquickcmp.dump.args_adapter import DumpArgsAdapter

# Test data
VALID_MODEL_PATH = "tests/model"
VALID_WEIGHT_PATH = "tests/weight"
VALID_INPUT_DATA_PATH = "tests/input"
VALID_CANN_PATH = "/usr/local/Ascend/ascend-toolkit/latest"
VALID_OUT_PATH = "tests/output"
VALID_INPUT_SHAPE = "input_shape"
VALID_DEVICE = "0"
VALID_DYM_SHAPE_RANGE = "dym_shape_range"
VALID_ONNX_FUSION_SWITCH = True
VALID_SAVED_MODEL_SIGNATURE = "saved_model_signature"
VALID_SAVED_MODEL_TAG_SET = "saved_model_tag_set"
VALID_DEVICE_PATTERN = "device_pattern"
VALID_TF_JSON_PATH = "tests/tf_json"
VALID_CUSTOM_OP = "custom_op"
VALID_DUMP = True
VALID_SINGLE_OP = "single_op"
VALID_OUTPUT_NODES = "output_nodes"

# Helper function to create directories and files
def create_test_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_test_file(path, content):
    with open(path, "w") as f:
        f.write(content)


# Test cases
def test_DumpArgsAdapter_given_valid_arguments_when_init_then_success():
    args = DumpArgsAdapter(
        model_path=VALID_MODEL_PATH,
        weight_path=VALID_WEIGHT_PATH,
        input_data_path=VALID_INPUT_DATA_PATH,
        cann_path=VALID_CANN_PATH,
        out_path=VALID_OUT_PATH,
        input_shape=VALID_INPUT_SHAPE,
        device=VALID_DEVICE,
        dym_shape_range=VALID_DYM_SHAPE_RANGE,
        onnx_fusion_switch=VALID_ONNX_FUSION_SWITCH,
        saved_model_signature=VALID_SAVED_MODEL_SIGNATURE,
        saved_model_tag_set=VALID_SAVED_MODEL_TAG_SET,
        device_pattern=VALID_DEVICE_PATTERN,
        tf_json_path=VALID_TF_JSON_PATH,
        custom_op=VALID_CUSTOM_OP,
        dump=VALID_DUMP,
        single_op=VALID_SINGLE_OP,
        output_nodes=VALID_OUTPUT_NODES,
    )
    assert args.model_path == VALID_MODEL_PATH
    assert args.weight_path == VALID_WEIGHT_PATH
    assert args.input_path == VALID_INPUT_DATA_PATH
    assert args.cann_path == VALID_CANN_PATH
    assert args.out_path == VALID_OUT_PATH
    assert args.input_shape == VALID_INPUT_SHAPE
    assert args.device == VALID_DEVICE
    assert args.dym_shape_range == VALID_DYM_SHAPE_RANGE
    assert args.onnx_fusion_switch == VALID_ONNX_FUSION_SWITCH
    assert args.saved_model_signature == VALID_SAVED_MODEL_SIGNATURE
    assert args.saved_model_tag_set == VALID_SAVED_MODEL_TAG_SET
    assert args.device_pattern == VALID_DEVICE_PATTERN
    assert args.tf_json_path == VALID_TF_JSON_PATH
    assert args.custom_op == VALID_CUSTOM_OP
    assert args.dump == VALID_DUMP
    assert args.single_op == VALID_SINGLE_OP
    assert args.output_nodes == VALID_OUTPUT_NODES


def test_DumpArgsAdapter_given_default_arguments_when_init_then_success():
    args = DumpArgsAdapter(model_path=VALID_MODEL_PATH)
    assert args.model_path == VALID_MODEL_PATH
    assert args.weight_path == ""
    assert args.input_path == ""
    assert args.out_path == "./"
    assert args.input_shape == ""
    assert args.device == "0"
    assert args.dym_shape_range == ""
    assert args.onnx_fusion_switch == True
    assert args.saved_model_signature == ""
    assert args.saved_model_tag_set == ""
    assert args.device_pattern == ""
    assert args.tf_json_path == ""
    assert args.custom_op == ""
    assert args.dump == True
    assert args.single_op == ""
    assert args.output_nodes == ""


def test_DumpArgsAdapter_given_empty_directory_when_init_then_success():
    create_test_dir(VALID_MODEL_PATH)
    create_test_dir(VALID_WEIGHT_PATH)
    create_test_dir(VALID_INPUT_DATA_PATH)
    create_test_dir(VALID_OUT_PATH)
    args = DumpArgsAdapter(
        model_path=VALID_MODEL_PATH,
        weight_path=VALID_WEIGHT_PATH,
        input_data_path=VALID_INPUT_DATA_PATH,
        cann_path=VALID_CANN_PATH,
        out_path=VALID_OUT_PATH,
    )
    assert args.model_path == VALID_MODEL_PATH
    assert args.weight_path == VALID_WEIGHT_PATH
    assert args.input_path == VALID_INPUT_DATA_PATH
    assert args.cann_path == VALID_CANN_PATH
    assert args.out_path == VALID_OUT_PATH
