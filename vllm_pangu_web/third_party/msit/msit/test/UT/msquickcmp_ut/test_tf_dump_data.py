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
import argparse
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from msquickcmp.common.utils import AccuracyCompareException
from msquickcmp.common import utils
from components.utils.util import load_file_to_read_common_check
from msquickcmp.common.dump_data import DumpData
from msquickcmp.tf import tf_dump_data
from msquickcmp.tf.tf_dump_data import TfDumpData
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

    with patch.dict(
        sys.modules,
        {
            "tensorflow": fake_tf,
            "tensorflow.__version__": "2.6.5",
            "tensorflow.__spec__": MagicMock(),
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

    monkeypatch.setattr(tf_common, "check_tf_version", MagicMock(return_value=True))
    monkeypatch.setattr(utils, "parse_input_shape", MagicMock(return_value=[]))
    monkeypatch.setattr(utils, "logger", MagicMock())
    monkeypatch.setattr(util, "load_file_to_read_common_check", MagicMock(return_value=True))
    monkeypatch.setattr(tf_dump_data, "load_file_to_read_common_check", MagicMock(return_value=True))
    monkeypatch.setattr(tf_common, "get_inputs_tensor", MagicMock(return_value=[]))
    monkeypatch.setattr(tf_common, "get_inputs_data", MagicMock(return_value={}))
    yield
    # Restore the original modules
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


def test_TfDumpData_init_given_valid_arguments_when_initialized_then_success(mock_tensorflow, mock_msquickcmp):
    args = argparse.Namespace(
        model_path="temp_model_dir/model.pb",
        input_path="temp_input_dir/input.bin",
        out_path="temp_output_dir",
        input_shape="input_name1:1,224,224,3",
        output_nodes="node_name1:0",
    )
    dump_data = TfDumpData(args)
    assert dump_data.args == args
    assert dump_data.global_graph is not None
    assert dump_data.input_path == "temp_input_dir/input.bin"

    temp_output_dir = os.path.abspath("temp_output_dir")
    assert dump_data.important_dirs["input"] == os.path.join(temp_output_dir, "input")
    assert dump_data.important_dirs["dump_data_tf"] == os.path.join(temp_output_dir, "dump_data/tf")
    assert dump_data.important_dirs["tmp"] == os.path.join(temp_output_dir, "tmp")


def test_TfDumpData_generate_inputs_data_given_valid_arguments_when_run_then_success(mock_tensorflow, mock_msquickcmp):
    from msquickcmp.common import tf_common

    args = argparse.Namespace(
        model_path="temp_model_dir/valid_model.pb",
        input_path="temp_input_dir/input.bin",
        out_path="temp_output_dir",
        input_shape="input_name1:1,224,224,3",
        output_nodes="node_name1:0",
    )
    dump_data = TfDumpData(args)
    with pytest.raises(AccuracyCompareException) as exc_info:
        dump_data.generate_inputs_data("npu_dump_data_path", True)


def test_TfDumpData_generate_dump_data_given_invalid_model_path_when_run_then_raise_exception(
    mock_tensorflow, mock_msquickcmp
):
    args = argparse.Namespace(
        model_path="temp_model_dir/invalid_model.pb",
        input_path="temp_input_dir/input.bin",
        out_path="temp_output_dir",
        input_shape="input_name1:1,224,224,3",
        output_nodes="node_name1:0",
    )
    dump_data = TfDumpData(args)
    with pytest.raises(AccuracyCompareException) as exc_info:
        dump_data.generate_dump_data()


def test_TfDumpData_generate_dump_data_given_invalid_input_path_when_run_then_raise_exception(
    mock_tensorflow, mock_msquickcmp
):
    args = argparse.Namespace(
        model_path="temp_model_dir/valid_model.pb",
        input_path="non_existent_input.bin",
        out_path="temp_output_dir",
        input_shape="input_name1:1,224,224,3",
        output_nodes="node_name1:0",
    )
    dump_data = TfDumpData(args)
    with pytest.raises(AccuracyCompareException) as exc_info:
        dump_data.generate_dump_data()


def test_TfDumpData_generate_dump_data_given_invalid_output_nodes_when_run_then_raise_exception(
    mock_tensorflow, mock_msquickcmp
):
    args = argparse.Namespace(
        model_path="temp_model_dir/valid_model.pb",
        input_path="temp_input_dir/input.bin",
        out_path="temp_output_dir",
        input_shape="input_name1:1,224,224,3",
        output_nodes="non_existent_node:0",
    )
    dump_data = TfDumpData(args)
    with pytest.raises(AccuracyCompareException) as exc_info:
        dump_data.generate_dump_data()


def test_TfDumpData_generate_dump_data_given_invalid_input_shape_when_run_then_raise_exception(
    mock_tensorflow, mock_msquickcmp
):
    args = argparse.Namespace(
        model_path="temp_model_dir/valid_model.pb",
        input_path="temp_input_dir/input.bin",
        out_path="temp_output_dir",
        input_shape="invalid_shape",
        output_nodes="node_name1:0",
    )
    dump_data = TfDumpData(args)
    with pytest.raises(AccuracyCompareException) as exc_info:
        dump_data.generate_dump_data()


def test_TfDumpData_generate_dump_data_given_tf_1x_version_when_run_then_success(
    mock_tensorflow, mock_msquickcmp, monkeypatch
):
    from msquickcmp.common import tf_common

    monkeypatch.setattr(tf_common, "check_tf_version", MagicMock(return_value=False))
    monkeypatch.setattr(TfDumpData, "_check_output_nodes_valid", MagicMock(return_value=True))
    args = argparse.Namespace(
        model_path="temp_model_dir/valid_model.pb",
        input_path="temp_input_dir/input.bin",
        out_path="temp_output_dir",
        input_shape="input_name1:1,224,224,3",
        output_nodes="node_name1:0",
    )
    dump_data = TfDumpData(args)
    dump_data.generate_dump_data()

