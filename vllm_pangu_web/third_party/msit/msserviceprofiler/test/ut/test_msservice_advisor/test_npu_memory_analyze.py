# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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
import json
import math
import pathlib
import subprocess
from collections import namedtuple
from unittest.mock import patch, MagicMock, mock_open

import pytest

# Import the module to test with proper error handling
from msserviceprofiler.msservice_advisor.profiling_analyze import npu_memory_analyze
from msserviceprofiler.msservice_advisor.profiling_analyze.register import REGISTRY, ANSWERS
from msserviceprofiler.msservice_advisor.profiling_analyze.utils import SUGGESTION_TYPES, BYTES_TO_GB
from msserviceprofiler.msguard import GlobalConfig


# Test fixtures
@pytest.fixture(autouse=True)
def reset_state():
    """Reset the REGISTRY and ANSWERS before each test"""
    REGISTRY.clear()
    for key in ANSWERS:
        ANSWERS[key].clear()
    yield


# Mock data
SAMPLE_SERVER_CONFIG = {
    "BackendConfig": {
        "ScheduleConfig": {"cacheBlockSize": 128, "maxIterTimes": 512, "supportSelectBatch": True},
        "npuDeviceIds": [[0, 1]],
        "ModelDeployConfig": {
            "ModelConfig": [{"npuMemSize": -1, "modelWeightPath": "/path/to/model", "tp": 2, "sp": 1}]
        },
    }
}

SAMPLE_MODEL_CONFIG = {"num_hidden_layers": 32, "num_attention_heads": 32, "hidden_size": 4096, "torch_dtype": "bf16"}

SAMPLE_BENCHMARK = {"result_perf": {"InputTokens": {"average": "100"}, "GeneratedTokens": {"average": "50"}}}


# Test get_benchmark_token_num
def test_get_benchmark_token_num_given_valid_input_returns_int():
    benchmark = {"result_perf": {"InputTokens": {"average": "100.5"}}}
    result = npu_memory_analyze.get_benchmark_token_num(benchmark, "InputTokens")
    assert result == 100
    assert isinstance(result, int)


def test_get_benchmark_token_num_given_missing_key_raises_exception():
    npu_memory_analyze.get_benchmark_token_num({}, "InputTokens")


# Test extract_token_num
def test_extract_token_num_given_benchmark_only_returns_benchmark_values():
    benchmark = {"result_perf": {"InputTokens": {"average": "100"}, "GeneratedTokens": {"average": "50"}}}
    input_params = MagicMock(input_token_num=0, output_token_num=0)
    input_tokens, output_tokens = npu_memory_analyze.extract_token_num(benchmark, input_params)
    assert input_tokens == 100
    assert output_tokens == 50


def test_extract_token_num_given_user_input_overrides_benchmark():
    benchmark = {"result_perf": {"InputTokens": {"average": "100"}, "GeneratedTokens": {"average": "50"}}}
    input_params = MagicMock(input_token_num=200, output_token_num=75)
    input_tokens, output_tokens = npu_memory_analyze.extract_token_num(benchmark, input_params)
    assert input_tokens == 200
    assert output_tokens == 75


# Test get_schedule_config_info
def test_get_schedule_config_info_given_valid_config_returns_values():
    backend_config = {"ScheduleConfig": {"cacheBlockSize": 128, "maxIterTimes": 512}}
    output_token_num, cache_block_sizes = npu_memory_analyze.get_schedule_config_info(backend_config, 0)
    assert output_token_num == 512
    assert cache_block_sizes == 128


def test_get_schedule_config_info_given_missing_cache_block_size_raises_exception():
    with pytest.raises(Exception):
        npu_memory_analyze.get_schedule_config_info({"ScheduleConfig": {}}, 0)


# Test get_model_config_info
def test_get_model_config_info_given_valid_config_returns_values():
    model_configs = {"tp": 2, "modelWeightPath": "/path/to/model", "sp": 1}
    npu_device_ids = [0, 1]
    tp, model_path = npu_memory_analyze.get_model_config_info(model_configs, 0, npu_device_ids)
    assert tp == 2
    assert model_path == "/path/to/model"


def test_get_model_config_info_given_missing_weight_path_raises_exception():
    with pytest.raises(Exception):
        npu_memory_analyze.get_model_config_info({}, 0, [0, 1])


# Test extract_server_config_params
def test_extract_server_config_params_given_valid_config_returns_params():
    args = MagicMock(output_token_num=0, tp=0)
    params = npu_memory_analyze.extract_server_config_params(SAMPLE_SERVER_CONFIG, args)


# Test extract_model_config_params
@patch("os.path.exists", return_value=True)
@patch("os.access", return_value=True)
@patch("os.path.isdir", return_value=True)
@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.stat", return_value=namedtuple("stat", ["st_size"])(1024))
@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(SAMPLE_MODEL_CONFIG))
def test_extract_model_config_params_given_valid_path_returns_params(*mocks):
    GlobalConfig.custom_return = True
    model_params, weight_size = npu_memory_analyze.extract_model_config_params("/path/to/model")
    GlobalConfig.reset()


# Test cal_npu_mem_size
@patch.object(npu_memory_analyze, "get_available_npu_memory", return_value=32.0)
def test_cal_npu_mem_size_given_positive_value_returns_value(mock_available):
    server_params = {"npu_mem_size": 16, "npu_device_ids": [0, 1], "model_weight_path": "/path/to/model"}
    result = npu_memory_analyze.cal_npu_mem_size(server_params, 10.0)
    assert result == 16


@patch.object(npu_memory_analyze, "get_available_npu_memory", return_value=32.0)
def test_cal_npu_mem_size_given_negative_value_calculates_value(mock_available):
    server_params = {"npu_mem_size": -1, "npu_device_ids": [0, 1], "model_weight_path": "/path/to/model"}
    result = npu_memory_analyze.cal_npu_mem_size(server_params, 10.0)
    assert result == math.floor((32.0 - 10.0 / 2) * 0.8)


# Test cal_total_block_num
def test_cal_total_block_num_given_valid_params_returns_blocks():
    server_params = {"tp": 2, "cache_block_sizes": 128, "sp": 1}
    model_params = {"num_hidden_layers": 32, "num_attention_heads": 32, "hidden_size": 4096, "torch_dtype": "bf16"}
    result = npu_memory_analyze.cal_total_block_num(16.0, server_params, model_params)
    assert isinstance(result, int)
    assert result > 0


# Test cal_block_nums
def test_cal_block_nums_given_valid_input_returns_blocks():
    max_block, min_block, avg_block = npu_memory_analyze.cal_block_nums(100, 128, 512, 50)
    assert max_block == math.ceil(100 / 128) + math.ceil(512 / 128)
    assert min_block == math.ceil(100 / 128)
    assert avg_block == math.ceil(100 / 128) + math.ceil(50 / 128)


# Test cal_max_batch_size_range
def test_cal_max_batch_size_range_given_valid_input_returns_batches():
    min_batch, max_batch, avg_batch = npu_memory_analyze.cal_max_batch_size_range(1000, 100, 50, 75, 1)
    assert min_batch == 1000 // math.ceil(100 / 1)
    assert max_batch == 1000 // math.ceil(50 / 1)
    assert avg_batch == 1000 // math.ceil(75 / 1)


# Test write_to_answer
def test_write_to_answer_given_valid_values_updates_answers():
    npu_memory_analyze.write_to_answer(10, 20, 15)
    assert "maxBatchSize" in ANSWERS[SUGGESTION_TYPES.config]
    assert "[10, 20]" in str(ANSWERS[SUGGESTION_TYPES.config]["maxBatchSize"])


# Test find_max_batch_size_range integration
@patch.object(npu_memory_analyze, "extract_token_num", return_value=(100, 50))
@patch.object(
    npu_memory_analyze,
    "extract_server_config_params",
    return_value={
        "output_token_num": 512,
        "cache_block_sizes": 128,
        "npu_device_ids": [0, 1],
        "tp": 2,
        "model_weight_path": "/path/to/model",
        "npu_mem_size": -1,
        "sp": 1,
    },
)
@patch.object(npu_memory_analyze, "extract_model_config_params", return_value=(SAMPLE_MODEL_CONFIG, 10.0))
@patch.object(npu_memory_analyze, "cal_npu_mem_size", return_value=16.0)
@patch.object(npu_memory_analyze, "cal_total_block_num", return_value=1000)
def test_find_max_batch_size_range_given_valid_input_updates_answers(
    mock_block, mock_mem, mock_model, mock_server, mock_token
):
    npu_memory_analyze.find_max_batch_size_range(SAMPLE_SERVER_CONFIG, SAMPLE_BENCHMARK, MagicMock())
    assert "maxBatchSize" in ANSWERS[SUGGESTION_TYPES.config]


# Test check_vaild_smi_output
def test_check_vaild_smi_output_given_valid_output_returns_header_and_colmap():
    sample_output = """
    NPU ID    Chip Logic ID    Some Other Column
    --------------------------------------------
    0         1000             Value1
    1         1001             Value2
    """
    header_line, col_map = npu_memory_analyze.check_vaild_smi_output(sample_output)
    assert "NPU ID" in header_line
    assert col_map["NPU ID"] == 0
    assert col_map["Chip Logic ID"] == 1


def test_check_vaild_smi_output_given_missing_columns_raises_error():
    with pytest.raises(ValueError, match="Missing required column"):
        npu_memory_analyze.check_vaild_smi_output("NPU ID    Wrong Column")


def test_check_vaild_smi_output_given_empty_output_raises_error():
    with pytest.raises(ValueError, match="npu-smi info -m output error"):
        npu_memory_analyze.check_vaild_smi_output("")


# Test get_npu_ids_map
@patch("subprocess.check_output")
def test_get_npu_ids_map_given_valid_output_returns_mapping(mock_subprocess):
    mock_output = """
    NPU ID    Chip Logic ID    Some Other Column
    --------------------------------------------
    0         1000             Value1
    1         1001             Value2
    2         1002             Value3
    """
    mock_subprocess.return_value = mock_output

    result = npu_memory_analyze.get_npu_ids_map()
    assert result == {1000: 0, 1001: 1, 1002: 2}
    mock_subprocess.assert_called_with(["npu-smi", "info", "-m"], text=True, stderr=subprocess.STDOUT)


@patch("subprocess.check_output")
def test_get_npu_ids_map_given_command_failure_raises_error(mock_subprocess):
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cmd")
    with pytest.raises(RuntimeError):
        npu_memory_analyze.get_npu_ids_map()


# Test cal_available_gm_memory
def test_cal_available_gm_memory_given_valid_values_returns_correct_memory():
    result = npu_memory_analyze.cal_available_gm_memory(32768, 50)  # 50% of 32GB
    assert result == 16.0  # (32768 * 0.5) / 1024


def test_cal_available_gm_memory_given_none_values_returns_zero():
    assert npu_memory_analyze.cal_available_gm_memory(None, 50) == 0
    assert npu_memory_analyze.cal_available_gm_memory(32768, None) == 0


# Test cal_npu_gm_memory
@patch("subprocess.check_output")
def test_cal_npu_gm_memory_given_valid_output_returns_memory(mock_subprocess):
    mock_output = """
    Some Header
    M Capacity(MB) : 32768
    M Usage Rate(%) : 25
    Other Lines
    """
    mock_subprocess.return_value = mock_output

    result = npu_memory_analyze.cal_npu_gm_memory(0)
    assert result == 24.0  # 32768 * 0.75 / 1024


@patch("subprocess.check_output")
def test_cal_npu_gm_memory_given_missing_values_returns_zero(mock_subprocess):
    mock_subprocess.return_value = "Invalid Output"
    result = npu_memory_analyze.cal_npu_gm_memory(0)
    assert result == 0


@patch("subprocess.check_output")
def test_cal_npu_gm_memory_given_command_failure_returns_zero(mock_subprocess):
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cmd")
    with patch.object(npu_memory_analyze.logger, "warning") as mock_log:
        result = npu_memory_analyze.cal_npu_gm_memory(0)
        assert result == 0
        mock_log.assert_called()


# Test get_available_npu_memory
@patch.object(npu_memory_analyze, "get_npu_ids_map")
@patch.object(npu_memory_analyze, "cal_npu_gm_memory")
def test_get_available_npu_memory_given_valid_input_returns_min_memory(mock_cal, mock_map):
    mock_map.return_value = {1000: 0, 1001: 1}
    mock_cal.side_effect = [16.0, 24.0]  # Different memory for different NPUs

    result = npu_memory_analyze.get_available_npu_memory([1000, 1001])
    assert result == 16.0  # Should return the minimum


@patch.object(npu_memory_analyze, "get_npu_ids_map")
def test_get_available_npu_memory_given_empty_map_raises_error(mock_map):
    mock_map.return_value = {}
    with pytest.raises(Exception, match="parse npu-smi info -m failed"):
        npu_memory_analyze.get_available_npu_memory([1000, 1001])


@patch.object(npu_memory_analyze, "get_npu_ids_map")
def test_get_available_npu_memory_given_missing_device_uses_zero(mock_map):
    mock_map.return_value = {1000: 0}
    with patch.object(npu_memory_analyze, "cal_npu_gm_memory", return_value=16.0):
        result = npu_memory_analyze.get_available_npu_memory([1000, 9999])  # 9999 not in map
        assert result == 16.0
