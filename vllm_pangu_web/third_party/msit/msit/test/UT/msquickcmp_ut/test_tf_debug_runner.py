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
import argparse
import shutil
from unittest.mock import MagicMock, patch

import pytest

from msquickcmp.common.utils import AccuracyCompareException
from msquickcmp.common import utils
from msquickcmp import tf_debug_runner
from msquickcmp.tf_debug_runner import TfDebugRunner
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
            "tensorflow.python": fake_tf.python,
            "tensorflow.python.debug": fake_tf.python.debug,
            "tensorflow.compat.v1.GraphDef": fake_tf.compat.v1.GraphDef,
            "tensorflow.compat.v1.Graph": fake_tf.compat.v1.Graph,
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
    monkeypatch.setattr(tf_debug_runner, "load_file_to_read_common_check", MagicMock(return_value=True))
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


def test_TfDebugRunner_init_given_valid_arguments_when_initialized_then_success():
    args = argparse.Namespace(
        model_path="temp_model_dir/model.pb",
        input_path="temp_input_dir/input.bin",
        out_path="temp_output_dir",
        input_shape="input_name1:1,224,224,3",
        output_nodes="node_name1:0",
    )

    runner = TfDebugRunner(args)
    assert runner.args == args
    assert runner.global_graph is None
    assert runner.input_shapes == {"input_name1": ["1", "224", "224", "3"]}
    assert runner.input_path == "temp_input_dir/input.bin"
    assert runner.dump_root == os.path.realpath("temp_output_dir")


def test_TfDebugRunner_run_given_valid_arguments_when_run_then_success(mock_tensorflow, mock_msquickcmp):
    args = argparse.Namespace(
        model_path="temp_model_dir/model.pb",
        input_path="temp_input_dir/input.bin",
        out_path="temp_output_dir",
        input_shape="input_name1:1,224,224,3",
        output_nodes="node_name1:0",
    )
    runner = TfDebugRunner(args)
    runner.run()
    # Assertions to check if the methods were called
    assert mock_tensorflow.io.gfile.GFile.called
    assert mock_tensorflow.compat.v1.GraphDef.FromString.called
    assert mock_tensorflow.compat.v1.Session.called

