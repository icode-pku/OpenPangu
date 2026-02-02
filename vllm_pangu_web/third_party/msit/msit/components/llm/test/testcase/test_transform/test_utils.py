import os
import string
import pytest
import torch
from unittest.mock import patch, MagicMock
from msit_llm.transform.utils import get_transform_scenario, SCENARIOS
from msit_llm.transform.torch_to_atb_python.utils import ATBModel
from msit_llm.transform.float_atb_to_quant_atb.utils import (
    check_libclang_so,
    filter_chinese_char,
    update_contents,
)


"""
msit_llm.transform.float_atb_to_quant_atb.utils
"""
# Mock logger to capture log messages
class MockLogger:
    def __init__(self):
        self.messages = []

    def debug(self, message):
        self.messages.append(("debug", message))

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

mock_logger = MockLogger()

# Mock the logger module
@pytest.fixture(autouse=True)
def mock_logger_module():
    with patch("msit_llm.common.log.logger", mock_logger):
        yield

# Test check_libclang_so
def test_check_libclang_so_given_libclang_so_exists_when_called_then_sets_library_file():
    with patch("os.path.exists", return_value=True), patch("clang.cindex.Config.set_library_file") as mock_set_library_file:
        check_libclang_so()
        mock_set_library_file.assert_called_once()

# Test filter_chinese_char
def test_filter_chinese_char_given_ascii_contents_when_called_then_returns_filtered_contents():
    contents = "abc123"
    assert filter_chinese_char(contents) == "abc123"

def test_filter_chinese_char_given_chinese_contents_when_called_then_returns_filtered_contents():
    contents = "你好世界"
    assert filter_chinese_char(contents) == ""

def test_filter_chinese_char_given_mixed_contents_when_called_then_returns_filtered_contents():
    contents = "你好abc世界"
    assert filter_chinese_char(contents) == "abc"

def test_filter_chinese_char_given_empty_contents_when_called_then_returns_filtered_contents():
    contents = ""
    assert filter_chinese_char(contents) == ""

# Test update_contents
def test_update_contents_given_valid_updates_when_called_then_returns_updated_contents():
    contents = "abcdef"
    updates = [(0, 0, "123"), (3, 3, "456")]
    assert update_contents(contents, updates) == "123abc456def"

def test_update_contents_given_empty_updates_when_called_then_returns_original_contents():
    contents = "abcdef"
    updates = []
    assert update_contents(contents, updates) == "abcdef"

def test_update_contents_given_updates_beyond_contents_when_called_then_returns_updated_contents():
    contents = "abcdef"
    updates = [(0, 10, "123")]
    assert update_contents(contents, updates) == "123"

"""
msit_llm.transform.utils
"""
# Mocking the necessary imports and functions
@pytest.fixture(autouse=True)
def mock_transformers():
    with patch('transformers.configuration_utils.PretrainedConfig.get_config_dict') as mock_get_config_dict:
        yield mock_get_config_dict

@pytest.fixture(autouse=True)
def mock_os_path():
    with patch('os.path.isfile') as mock_isfile, patch('os.path.isdir') as mock_isdir, patch('os.listdir') as mock_listdir:
        yield mock_isfile, mock_isdir, mock_listdir

# Test cases
def test_get_transform_scenario_given_single_cpp_file_when_valid_path_then_float_atb_to_quant_atb(mock_os_path):
    mock_isfile, _, _ = mock_os_path
    mock_isfile.return_value = True
    assert get_transform_scenario("path/to/file.cpp") == SCENARIOS.float_atb_to_quant_atb

#
def test_get_transform_scenario_given_pretrained_config_file_when_valid_path_then_torch_to_float_python_atb(mock_os_path, mock_transformers):
    mock_isfile, _, _ = mock_os_path

    mock_isfile.return_value = True
    mock_transformers.return_value = {}
    assert get_transform_scenario("path/to/config.json", to_python=True) == SCENARIOS.torch_to_float_python_atb

#
def test_get_transform_scenario_given_directory_with_cpp_files_when_valid_path_then_float_atb_to_quant_atb(mock_os_path):
    mock_isfile, mock_isdir, mock_listdir = mock_os_path
    mock_isfile.return_value = False
    mock_isdir.return_value = True
    mock_listdir.return_value = ["file1.cpp", "file2.cpp"]
    assert get_transform_scenario("path/to/cpp_directory") == SCENARIOS.torch_to_float_atb

#
def test_get_transform_scenario_given_invalid_input_when_empty_directory_then_none(mock_os_path):
    mock_isfile, mock_isdir, mock_listdir = mock_os_path
    mock_isfile.return_value = False
    mock_isdir.return_value = True
    mock_listdir.return_value = []
    assert get_transform_scenario("path/to/empty_directory") == SCENARIOS.torch_to_float_atb

#
def test_get_transform_scenario_given_non_existent_path_when_invalid_path_then_none(mock_os_path):
    mock_isfile, mock_isdir, _ = mock_os_path
    mock_isfile.return_value = False
    mock_isdir.return_value = False
    assert get_transform_scenario("path/to/non_existent_file.cpp") == SCENARIOS.torch_to_float_atb

#
def test_get_transform_scenario_given_empty_string_path_when_empty_path_then_none():
    assert get_transform_scenario("") == SCENARIOS.torch_to_float_atb

def test_get_transform_scenario_given_path_with_special_chars_when_valid_path_then_float_atb_to_quant_atb(mock_os_path):
    mock_isfile, _, _ = mock_os_path
    mock_isfile.return_value = True
    assert get_transform_scenario("path/to/file_with_special_chars!@#$%^&*().cpp") == SCENARIOS.float_atb_to_quant_atb

def test_get_transform_scenario_given_path_with_non_ascii_chars_when_valid_path_then_float_atb_to_quant_atb(mock_os_path):
    mock_isfile, _, _ = mock_os_path
    mock_isfile.return_value = True
    assert get_transform_scenario("path/to/file_with_non_ascii_字符.cpp") == SCENARIOS.float_atb_to_quant_atb



"""
msit_llm.transform.torch_to_atb_python.utils
"""
# Mock the necessary dependencies
class MockATBModel:
    def __init__(self):
        self.input_names = ["input_ids", "position_ids", "inputs_embeds"]
        self.output_names = ["output"]
        self.head_dim = 64
        self.num_attention_heads = 12
        self.num_key_value_heads = 12
        self.vocab_size = 50257
        self.rope_theta = 1e4

    def forward(self, inputs, outputs, bind_map):
        return outputs

    def set_weights(self, weights):
        pass

@pytest.fixture
def mock_atb_model():
    return MockATBModel()

@pytest.fixture
def atb_model(mock_atb_model):
    return ATBModel(mock_atb_model, atb_model_config={"max_seq_len": 1024, "max_batch_size": 1})

# Test cases
def test_atb_model_given_valid_input_ids_when_forward_then_success(atb_model):
    input_ids = torch.tensor([[1, 2, 3]])
    output = atb_model(input_ids=input_ids)
    assert output is not None

def test_atb_model_given_missing_inputs_embeds_when_forward_then_success(atb_model):
    input_ids = torch.tensor([[1, 2, 3]])
    output = atb_model(input_ids=input_ids)
    assert output is not None

def test_atb_model_given_invalid_inputs_embeds_shape_when_forward_then_fail(atb_model):
    inputs_embeds = torch.randn(1, 64)
    with pytest.raises(Exception):
        atb_model(inputs_embeds=inputs_embeds)

def test_atb_model_given_invalid_inputs_embeds_dtype_when_forward_then_fail(atb_model):
    inputs_embeds = torch.randint(0, 10, (1, 3, 64))
    with pytest.raises(Exception):
        atb_model(inputs_embeds=inputs_embeds)

def test_atb_model_given_large_input_ids_when_forward_then_success(atb_model):
    input_ids = torch.randint(0, 10000, (10, 1024))
    output = atb_model(input_ids=input_ids)
    assert output is not None

def test_atb_model_given_empty_input_ids_when_forward_then_success(atb_model):
    input_ids = torch.tensor([])
    output = atb_model(input_ids=input_ids)
    assert output is not None

def test_atb_model_given_none_input_ids_when_forward_then_success(atb_model):
    output = atb_model(input_ids=None)
    assert output is not None

def test_atb_model_given_large_input_values_when_forward_then_success(atb_model):
    input_ids = torch.tensor([[1000000, 2000000, 3000000]])
    output = atb_model(input_ids=input_ids)
    assert output is not None

def test_atb_model_given_negative_input_values_when_forward_then_success(atb_model):
    input_ids = torch.tensor([[-1, -2, -3]])
    output = atb_model(input_ids=input_ids)
    assert output is not None

def test_atb_model_given_mixed_data_types_when_forward_then_fail(atb_model):
    input_ids = torch.tensor([[1, 2, 3]])
    inputs_embeds = torch.randint(0, 10, (1, 3, 64))
    with pytest.raises(Exception):
        atb_model(input_ids=input_ids, inputs_embeds=inputs_embeds)