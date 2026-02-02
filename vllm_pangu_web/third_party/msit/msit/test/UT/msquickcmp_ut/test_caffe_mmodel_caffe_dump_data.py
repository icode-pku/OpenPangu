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
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the class to be tested
from msquickcmp.caffe_model.caffe_dump_data import CaffeDumpData
from msquickcmp.common.utils import AccuracyCompareException

# Fixture to create and clean up temporary directories
@pytest.fixture(scope="function", autouse=True)
def temp_dir():
    temp_dir = "temp_test_dir"
    os.makedirs(temp_dir, exist_ok=True)
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


# Mock the caffe module if not installed
@pytest.fixture(scope="module", autouse=True)
def mock_caffe():
    try:
        import caffe
    except ImportError:
        caffe = MagicMock()
        caffe.Net = MagicMock()
        caffe.TEST = MagicMock()
        sys.modules["caffe"] = caffe
    yield
    if "caffe" in sys.modules and isinstance(sys.modules["caffe"], MagicMock):
        del sys.modules["caffe"]


# Mock the tensorflow module if not installed
@pytest.fixture(scope="module", autouse=True)
def mock_tensorflow():
    try:
        import tensorflow
    except ImportError:
        tensorflow = MagicMock()
        sys.modules["tensorflow"] = tensorflow
    yield
    if "tensorflow" in sys.modules and isinstance(sys.modules["tensorflow"], MagicMock):
        del sys.modules["tensorflow"]


# Test valid inputs
def test_caffe_dump_data_given_valid_paths_when_initialized_then_pass(temp_dir):
    arguments = MagicMock()
    arguments.model_path = os.path.join(temp_dir, "valid_model.prototxt")
    arguments.weight_path = os.path.join(temp_dir, "valid_weights.caffemodel")
    arguments.out_path = os.path.join(temp_dir, "output")
    arguments.input_path = None
    arguments.input_shape = ""

    with patch("msquickcmp.common.utils.create_directory"), patch("msquickcmp.common.utils.logger.warning"), patch(
        "msquickcmp.common.dump_data.DumpData._check_path_exists"
    ), patch("msquickcmp.common.utils.logger.info"):
        caffe_dump_data = CaffeDumpData(arguments)
        assert caffe_dump_data.model_path == os.path.realpath(arguments.model_path)
        assert caffe_dump_data.weight_path == os.path.realpath(arguments.weight_path)
        assert caffe_dump_data.output_path == os.path.realpath(arguments.out_path)


# Test invalid model path
def test_caffe_dump_data_given_invalid_model_path_when_initialized_then_fail(temp_dir):
    arguments = MagicMock()
    arguments.model_path = os.path.join(temp_dir, "invalid_model.prototxt")
    arguments.weight_path = os.path.join(temp_dir, "valid_weights.caffemodel")
    arguments.out_path = os.path.join(temp_dir, "output")
    arguments.input_path = None
    arguments.input_shape = ""

    with pytest.raises(AccuracyCompareException):
        CaffeDumpData(arguments)


# Test invalid weight path
def test_caffe_dump_data_given_invalid_weight_path_when_initialized_then_fail(temp_dir):
    arguments = MagicMock()
    arguments.model_path = os.path.join(temp_dir, "valid_model.prototxt")
    arguments.weight_path = os.path.join(temp_dir, "invalid_weights.caffemodel")
    arguments.out_path = os.path.join(temp_dir, "output")
    arguments.input_path = None
    arguments.input_shape = ""

    with pytest.raises(AccuracyCompareException):
        CaffeDumpData(arguments)


# Test dynamic input shapes warning
def test_caffe_dump_data_given_dynamic_input_shapes_when_initialized_then_warn(temp_dir, caplog):
    arguments = MagicMock()
    arguments.model_path = os.path.join(temp_dir, "valid_model.prototxt")
    arguments.weight_path = os.path.join(temp_dir, "valid_weights.caffemodel")
    arguments.out_path = os.path.join(temp_dir, "output")
    arguments.input_path = None
    arguments.input_shape = "1,3,224,224"

    with patch("msquickcmp.common.utils.create_directory"), patch("msquickcmp.common.utils.logger.warning"), patch(
        "msquickcmp.common.dump_data.DumpData._check_path_exists"
    ), patch("msquickcmp.common.utils._check_colon_exist"), patch("msquickcmp.common.utils.logger.info"):
        with pytest.raises(AccuracyCompareException):
            CaffeDumpData(arguments)
