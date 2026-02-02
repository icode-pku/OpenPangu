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
import unittest
from unittest.mock import patch, MagicMock

import pytest
import numpy as np


# Mocking TensorFlow
class MockTensorFlow:
    __version__ = "2.0.0"
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    int64 = "int64"
    int32 = "int32"
    int16 = "int16"
    int8 = "int8"
    uint8 = "uint8"
    bool = "bool"
    complex64 = "complex64"
    dtypes = MagicMock()
    compat = MagicMock()
    saved_model = MagicMock()
    Session = MagicMock()


@pytest.fixture(scope="function")
def import_tf_common():
    backup = {}
    for mod in ['tensorflow', 'msquickcmp.common']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
            
    sys.modules['tensorflow'] = MockTensorFlow
    from msquickcmp.common import tf_common
    yield tf_common
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['tensorflow', 'msquickcmp.common']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    # Setup
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/test_file.bin", "wb") as f:
        f.write(np.array([1, 2, 3], dtype=np.float32).tobytes())

    yield

    # Teardown
    if os.path.exists("test_dir"):
        os.remove("test_dir/test_file.bin")
        os.rmdir("test_dir")


def test_execute_command_given_valid_cmd_when_executed_then_pass(import_tf_common):
    tf_common = import_tf_common

    cmd = ["echo", "test"]
    status_code = tf_common.execute_command(cmd)
    assert status_code == 0


def test_execute_command_given_none_cmd_when_executed_then_fail(import_tf_common):
    tf_common = import_tf_common

    cmd = None
    status_code = tf_common.execute_command(cmd)
    assert status_code == -1


def test_convert_to_numpy_type_given_valid_tensor_type_when_converted_then_pass(import_tf_common):
    tf_common = import_tf_common

    tensor_type = MockTensorFlow.float32
    np_type = tf_common.convert_to_numpy_type(tensor_type)
    assert np_type == np.float32


def test_convert_to_numpy_type_given_invalid_tensor_type_when_converted_then_fail(import_tf_common):
    tf_common = import_tf_common

    tensor_type = "invalid_type"
    with pytest.raises(tf_common.utils.AccuracyCompareException):
        tf_common.convert_to_numpy_type(tensor_type)


def test_convert_tensor_shape_given_valid_tensor_shape_when_converted_then_pass(import_tf_common):
    tf_common = import_tf_common

    tensor_shape = MagicMock()
    tensor_shape.as_list.return_value = [1, 2, 3]
    shape_tuple = tf_common.convert_tensor_shape(tensor_shape)
    assert shape_tuple == (1, 2, 3)


def test_convert_tensor_shape_given_none_dim_when_converted_then_fail(import_tf_common):
    tf_common = import_tf_common

    tensor_shape = MagicMock()
    tensor_shape.as_list.return_value = [1, None, 3]
    with pytest.raises(tf_common.utils.AccuracyCompareException):
        tf_common.convert_tensor_shape(tensor_shape)


def test_verify_and_adapt_dynamic_shape_given_valid_input_shapes_when_verified_then_pass(import_tf_common):
    tf_common = import_tf_common

    input_shapes = {"op_name": [1, 2, 3]}
    op_name = "op_name"
    tensor = MagicMock()
    tensor.shape = [1, None, 3]
    adapted_tensor = tf_common.verify_and_adapt_dynamic_shape(input_shapes, op_name, tensor)
    assert adapted_tensor.shape == [1, None, 3]


def test_verify_and_adapt_dynamic_shape_given_invalid_input_shapes_when_verified_then_fail(import_tf_common):
    tf_common = import_tf_common

    input_shapes = {"op_name": [1, 2, 4]}
    op_name = "op_name"
    tensor = MagicMock()
    tensor.shape = [1, None, 3]
    with pytest.raises(tf_common.utils.AccuracyCompareException):
        tf_common.verify_and_adapt_dynamic_shape(input_shapes, op_name, tensor)


def test_get_inputs_data_given_valid_input_paths_when_retrieved_then_pass(import_tf_common):
    tf_common = import_tf_common

    inputs_tensor = [MagicMock(dtype=MockTensorFlow.float32, shape=[3])]
    input_paths = "test_dir/test_file.bin"
    inputs_map = tf_common.get_inputs_data(inputs_tensor, input_paths)
    assert len(inputs_map) == 1


def test_get_model_inputs_dtype_given_valid_model_path_when_retrieved_then_pass(import_tf_common):
    tf_common = import_tf_common

    model_path = "test_dir"
    serving = "serving_default"
    tag_set = ""
    inputs_dtype = tf_common.get_model_inputs_dtype(model_path, serving, tag_set)
    assert isinstance(inputs_dtype, dict)


def test_split_tag_set_given_valid_tag_set_when_split_then_pass(import_tf_common):
    tf_common = import_tf_common

    saved_model_tag_set = "tag1,tag2"
    tag_sets = tf_common.split_tag_set(saved_model_tag_set)
    assert tag_sets == ["tag1", "tag2"]


def test_load_file_to_read_common_check_with_walk_given_valid_model_path_when_checked_then_pass(import_tf_common):
    tf_common = import_tf_common

    model_path = "test_dir"
    tf_common.load_file_to_read_common_check_with_walk(model_path)



class TestCheckTFVersion(unittest.TestCase):

    def setUp(self):
        self.original_tensorflow = sys.modules.get("tensorflow", None)
        sys.modules["tensorflow"] = MockTensorFlow()

        from components.debug.compare.msquickcmp.common import tf_common
        self.tf_common = tf_common

    def tearDown(self):
        if self.original_tensorflow is None:
            del sys.modules["tensorflow"]
        else:
            sys.modules["tensorflow"] = self.original_tensorflow

    def test_version_match(self):
        self.tf_common.tf.__version__ = "2.10.0"
        self.assertTrue(self.tf_common.check_tf_version("2"))

    def test_version_not_match(self):
        self.tf_common.tf.__version__ = "1.15.0"
        self.assertFalse(self.tf_common.check_tf_version("2"))


class TestGetInputsTensor(unittest.TestCase):

    @patch("components.debug.compare.msquickcmp.common.tf_common.utils")
    def test_get_inputs_tensor(self, mock_utils):
        mock_utils.parse_input_shape.return_value = ["input_tensor"]
        mock_utils.logger = MagicMock()
        mock_utils.check_input_name_in_model = MagicMock()
        mock_graph = MagicMock()

        mock_op1 = MagicMock()
        mock_op1.name = "input_tensor"
        mock_op1.type = "Placeholder"

        mock_op2 = MagicMock()
        mock_op2.name = "other_tensor"
        mock_op2.type = "NotPlaceholder"
        mock_graph.get_operations.return_value = [mock_op1, mock_op2]
        mock_tensor = MagicMock()
        mock_graph.get_tensor_by_name.return_value = mock_tensor

        with patch("components.debug.compare.msquickcmp.common.tf_common.verify_and_adapt_dynamic_shape") as mock_verify:
            mock_verify.return_value = "mock_adapted_tensor"
            from components.debug.compare.msquickcmp.common.tf_common import get_inputs_tensor
            result = get_inputs_tensor(mock_graph, "shape_string")

            mock_utils.parse_input_shape.assert_called_once_with("shape_string")
            mock_utils.check_input_name_in_model.assert_called_once()
            mock_graph.get_tensor_by_name.assert_called_once_with("input_tensor:0")
            mock_verify.assert_called_once()

            self.assertEqual(result, ["mock_adapted_tensor"])
