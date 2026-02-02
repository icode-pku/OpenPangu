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
import os
import sys
import shutil
import json
import argparse
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from msquickcmp.common.utils import AccuracyCompareException
from msquickcmp.common import utils
from components.utils.util import load_file_to_read_common_check
from components.utils.check.rule import Rule
from msquickcmp.tf import tf_save_model_dump_data
from msquickcmp.tf.tf_save_model_dump_data import TfSaveModelDumpData, parse_ops_name_from_om_json
from components.utils import util


# Mocking the necessary modules
@pytest.fixture
def mock_tensorflow(monkeypatch):
    # Create a fake tensorflow module
    fake_tf = MagicMock()
    fake_tf.io.gfile.GFile = MagicMock()
    fake_tf.compat.v1.GraphDef = MagicMock()
    fake_tf.compat.v1.Graph = MagicMock()
    fake_tf.compat.v1.Session = MagicMock()
    fake_tf.compat.v1.ConfigProto = MagicMock()
    fake_tf.python.debug = MagicMock()
    fake_tf.python.debug.LocalCLIDebugWrapperSession = MagicMock()
    fake_tf.dtypes.as_dtype = MagicMock(return_value=MagicMock(as_numpy_dtype=np.float32))
    fake_tf.saved_model.tag_constants.SERVING = "serve"

    # Use a context manager to patch the modules
    with patch.dict(
        sys.modules,
        {
            "tensorflow": fake_tf,
            "tensorflow.__version__": "2.6.5",
            "tensorflow.python": fake_tf.python,
            "tensorflow.python.debug": fake_tf.python.debug,
            "tensorflow.io.gfile.GFile": fake_tf.io.gfile.GFile,
            "tfdbg_ascend": MagicMock(),
        },
    ):
        yield fake_tf


@pytest.fixture
def mock_msquickcmp(monkeypatch):
    from msquickcmp.common import tf_common

    monkeypatch.setattr(TfSaveModelDumpData, "_check_tf_version", MagicMock(return_value=True))
    monkeypatch.setattr(tf_common, "check_tf_version", MagicMock(return_value=True))
    monkeypatch.setattr(utils, "parse_input_shape", MagicMock(return_value={}))
    monkeypatch.setattr(utils, "logger", MagicMock())
    monkeypatch.setattr(util, "load_file_to_read_common_check", MagicMock(return_value=True))
    monkeypatch.setattr(tf_save_model_dump_data, "load_file_to_read_common_check", MagicMock(return_value=True))
    monkeypatch.setattr(tf_common, "load_file_to_read_common_check", MagicMock(return_value=True))
    monkeypatch.setattr(tf_common, "get_inputs_tensor", MagicMock(return_value=[]))
    monkeypatch.setattr(tf_common, "get_inputs_data", MagicMock(return_value={}))
    monkeypatch.setattr(tf_common, "execute_command", MagicMock(return_value=0))
    yield
    # Restore the original modules
    monkeypatch.undo()


@pytest.fixture(scope="function", autouse=True)
def mock_load_file_to_read_common_check(monkeypatch):
    def mock_load_file(path):
        if "valid" in path:
            return path
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_OPEN_FILE_ERROR)

    monkeypatch.setattr("components.utils.util.load_file_to_read_common_check", mock_load_file)
    yield
    # Restore the original function
    monkeypatch.undo()


@pytest.fixture(scope="module", autouse=True)
def setup_teardown_files():
    # Create temporary files and directories
    os.makedirs("temp_model_dir", exist_ok=True)
    os.makedirs("temp_input_dir", exist_ok=True)
    os.makedirs("temp_output_dir", exist_ok=True)
    yield
    # Clean up temporary files and directories
    shutil.rmtree("temp_model_dir")
    shutil.rmtree("temp_input_dir")
    shutil.rmtree("temp_output_dir")


def test_parse_ops_name_from_om_json_given_valid_json_path_when_parsed_then_return_op_names(
    mock_tensorflow, mock_msquickcmp
):
    tf_json_path = "temp_model_dir/valid_model.json"
    with open(tf_json_path, "w") as f:
        json.dump(
            {
                "graph": [
                    {"op": [{"output_desc": [{"attr": [{"key": "_datadump_origin_name", "value": {"s": "op1"}}]}]}]}
                ]
            },
            f,
        )
    op_names = parse_ops_name_from_om_json(tf_json_path)
    assert op_names == ["op1"]


def test_TfSaveModelDumpData_init_given_valid_arguments_when_initialized_then_success(mock_tensorflow, mock_msquickcmp):
    args = argparse.Namespace(
        out_path="temp_output_dir",
        saved_model_signature="serving_default",
        saved_model_tag_set="serve",
        input_path="temp_input_dir/input.bin",
        input_shape="input_name:1,224,224,3",
    )
    model_path = "temp_model_dir/valid_model.pb"
    dump_data = TfSaveModelDumpData(args, model_path)
    temp_output_dir = os.path.abspath("temp_output_dir")
    assert dump_data.serving == "serving_default"
    assert dump_data.tag_set == {"serve"}
    assert dump_data.input == os.path.join(temp_output_dir, "input")
    assert dump_data.dump_data_tf == os.path.join(temp_output_dir, "dump_data", "tf")
    assert dump_data.input_path == os.path.join("temp_input_dir", "input.bin")
    assert dump_data.model_path == model_path
    assert dump_data.input_shape_list == [("input_name", [1, 224, 224, 3])]


def test_TfSaveModelDumpData_generate_inputs_data_for_dump_given_valid_input_path_when_run_then_success(
    mock_tensorflow, mock_msquickcmp
):
    args = argparse.Namespace(
        out_path="temp_output_dir",
        saved_model_signature="serving_default",
        saved_model_tag_set="serve",
        input_path="temp_input_dir/input.bin",
        input_shape="input_name:1,224,224,3",
    )
    model_path = "temp_model_dir/valid_model.pb"
    dump_data = TfSaveModelDumpData(args, model_path)

    with pytest.raises(AccuracyCompareException) as exc_info:
        dump_data.generate_inputs_data_for_dump()


def test_TfSaveModelDumpData_generate_inputs_data_for_dump_given_no_input_path_when_run_then_success(
    mock_tensorflow, mock_msquickcmp
):
    args = argparse.Namespace(
        out_path="temp_output_dir",
        saved_model_signature="serving_default",
        saved_model_tag_set="serve",
        input_path="",
        input_shape="input_name:1,224,224,3",
    )
    model_path = "temp_model_dir/valid_model.pb"
    dump_data = TfSaveModelDumpData(args, model_path)
    dump_data.generate_inputs_data_for_dump()
    assert "input_name" in dump_data.inputs_data


def test_TfSaveModelDumpData_generate_dump_data_given_valid_tf_json_path_when_run_then_success(
    mock_tensorflow, mock_msquickcmp
):
    args = argparse.Namespace(
        out_path="temp_output_dir",
        saved_model_signature="serving_default",
        saved_model_tag_set="serve",
        input_path="temp_input_dir/input.bin",
        input_shape="input_name:1,224,224,3",
    )
    model_path = "temp_model_dir/valid_model.pb"
    dump_data = TfSaveModelDumpData(args, model_path)
    tf_json_path = "temp_model_dir/valid_model.json"
    with open(tf_json_path, "w") as f:
        json.dump(
            {
                "graph": [
                    {"op": [{"output_desc": [{"attr": [{"key": "_datadump_origin_name", "value": {"s": "op1"}}]}]}]}
                ]
            },
            f,
        )
    with pytest.raises(ValueError) as exc_info:
        dump_data.generate_dump_data(tf_json_path)


def test_TfSaveModelDumpData_generate_dump_data_given_invalid_tf_json_path_when_run_then_raise_exception(
    mock_tensorflow, mock_msquickcmp
):
    args = argparse.Namespace(
        out_path="temp_output_dir",
        saved_model_signature="serving_default",
        saved_model_tag_set="serve",
        input_path="temp_input_dir/input.bin",
        input_shape="input_name:1,224,224,3",
    )
    model_path = "temp_model_dir/valid_model.pb"
    dump_data = TfSaveModelDumpData(args, model_path)
    tf_json_path = "temp_model_dir/invalid_model.json"
    with pytest.raises(FileNotFoundError):
        dump_data.generate_dump_data(tf_json_path)
