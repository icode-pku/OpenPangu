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
import argparse
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import asdict
import pytest

from msserviceprofiler.msservice_advisor import advisor
from msserviceprofiler.msservice_advisor.profiling_analyze.utils import TARGETS, SUGGESTION_TYPES, logger
from msserviceprofiler.msservice_advisor.profiling_analyze import utils
from msserviceprofiler.msguard import GlobalConfig


# Test fixtures
@pytest.fixture(autouse=True)
def reset_state():
    """Reset environment variables before each test"""
    if advisor.MIES_INSTALL_PATH in os.environ:
        del os.environ[advisor.MIES_INSTALL_PATH]
    yield


# Test data
SAMPLE_REQ_TO_DATA_MAP = {"21559056a7ff44c88a891ecbb537c431": "0"}
SAMPLE_RESULT_PERF = "FirstTokenTime,DecodeTime\naverage,max\n213.2031,228.3775"
SAMPLE_RESULT_COMMON = "Concurrency,ModelName\n50,DeepSeek-R1"
SAMPLE_RESULTS_PER_REQUEST = {"7": {"input_len": 213, "output_len": 12}}
SAMPLE_CONFIG_JSON = {"BackendConfig": {"ModelDeployConfig": {}}, "LogConfig": {"logPath": "custom_logs/server.log"}}


# Test ProfilingParameters
def test_profiling_parameters_extract_from_args():
    args = MagicMock(target="ttft", target_metrics="average", input_token_num=100, output_token_num=50, tp=2)
    params = advisor.ProfilingParameters.extract_from_args(args)
    assert params.target == "ttft"
    assert params.input_token_num == 100
    assert params.tp == 2


# Test check_positive_integer
def test_check_positive_integer_given_valid_input():
    assert advisor.check_positive_integer("10") == 10
    assert advisor.check_positive_integer(5) == 5


def test_check_positive_integer_given_invalid_input():
    with pytest.raises(ValueError):
        advisor.check_positive_integer("-1")
    with pytest.raises(ValueError):
        advisor.check_positive_integer("abc")


# Test get_latest_matching_file
@patch("glob.glob")
def test_get_latest_matching_file_returns_none_if_no_files(mock_glob):
    mock_glob.return_value = []
    assert advisor.get_latest_matching_file("/path", "pattern") is None


@patch.object(utils, "read_csv")
@patch.object(utils, "read_json")
def test_read_csv_or_json_dispatches_correctly(mock_read_json, mock_read_csv):
    GlobalConfig.custom_return = True
    # Mock the file path and extension checking
    with patch("os.path.exists", return_value=True):
        # Test CSV file
        utils.read_csv_or_json("file.csv")
        mock_read_csv.assert_called_once_with("file.csv")
        mock_read_json.assert_not_called()

        # Reset mocks for next test
        mock_read_csv.reset_mock()
        mock_read_json.reset_mock()

        # Test JSON file
        utils.read_csv_or_json("file.json")
        mock_read_json.assert_called_once_with("file.json")
        mock_read_csv.assert_not_called()

        # Reset mocks for next test
        mock_read_csv.reset_mock()
        mock_read_json.reset_mock()

        # Test unknown extension
        assert utils.read_csv_or_json("file.txt") is None
        mock_read_csv.assert_not_called()
        mock_read_json.assert_not_called()

        # Test non-existent file
        with patch("os.path.exists", return_value=False):
            assert utils.read_csv_or_json("nonexistent.csv") is None
            mock_read_csv.assert_not_called()
            mock_read_json.assert_not_called()
    GlobalConfig.reset()


# Test parse_benchmark_instance
@patch.object(advisor, "get_latest_matching_file")
@patch.object(utils, "read_csv_or_json")
def test_parse_benchmark_instance(mock_read, mock_latest):
    # Setup mock returns
    GlobalConfig.custom_return = True
    mock_latest.side_effect = ["req_map.json", "result_perf.csv", "result_common.csv", "results_per_request.json"]
    mock_read.side_effect = [
        SAMPLE_REQ_TO_DATA_MAP,
        {"FirstTokenTime": ["average", "max"], "DecodeTime": ["213.2031", "228.3775"]},
        {"Concurrency": ["50"], "ModelName": ["DeepSeek-R1"]},
        SAMPLE_RESULTS_PER_REQUEST,
    ]

    with patch.object(advisor.logger, "debug"):
        result = advisor.parse_benchmark_instance("/path")
    GlobalConfig.reset()


# Test parse_mindie_server_config
def test_parse_mindie_server_config_with_json_path():
    with patch.object(utils, "read_csv_or_json", return_value=SAMPLE_CONFIG_JSON):
        config = advisor.parse_mindie_server_config("/path/config.json")


def test_parse_mindie_server_config_with_service_path():
    with patch.object(utils, "read_csv_or_json", return_value=SAMPLE_CONFIG_JSON):
        config = advisor.parse_mindie_server_config("/service/path")


# Test analyze
@patch("msserviceprofiler.msservice_advisor.profiling_analyze.register.REGISTRY", {"test_analyzer": MagicMock()})
@patch(
    "msserviceprofiler.msservice_advisor.profiling_analyze.register.ANSWERS",
    {"config": {"param": [("action", "reason")]}},
)
def test_analyze_calls_registered_analyzers():
    params = advisor.ProfilingParameters(
        target="ttft", target_metrics="average", input_token_num=100, output_token_num=50, tp=2
    )

    with patch.object(advisor.logger, "info") as mock_log:
        advisor.analyze({}, {}, params)

        # Verify analyzer was called
        assert mock_log.call_count >= 3


# Test arg_parse
def test_arg_parse_sets_up_parser_correctly():
    subparsers = MagicMock()
    advisor.arg_parse(subparsers)

    # Verify parser was created with correct arguments
    subparsers.add_parser.assert_called_once_with(
        "advisor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="advisor for MindIE Service performance"
    )
    parser = subparsers.add_parser.return_value
    assert parser.add_argument.call_count == 8


# Test main
@patch.object(advisor, "parse_benchmark_instance")
@patch.object(advisor, "parse_mindie_server_config")
@patch.object(advisor, "analyze")
@patch.object(advisor, "set_log_level")
def test_main_integration(mock_log_level, mock_analyze, mock_parse_config, mock_parse_benchmark):
    # Setup test args
    args = MagicMock(
        instance_path="instance",
        service_config_path="config.json",
        target="ttft",
        target_metrics="average",
        input_token_num=100,
        output_token_num=50,
        tp=2,
        log_level="info",
    )

    # Setup mocks
    mock_parse_benchmark.return_value = {"benchmark": "data"}
    mock_parse_config.return_value = ({"config": "data"})

    advisor.main(args)

    # Verify all components were called
    mock_log_level.assert_called_with("info")
    mock_parse_benchmark.assert_called_with("instance")
    mock_parse_config.assert_called_with("config.json")
    mock_analyze.assert_called()


# Test arg_parse
def test_arg_parse_with_actual_parsing():
    # Create actual parser and subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Add our advisor subparser
    advisor.arg_parse(subparsers)

    # Test with minimal required arguments
    args = parser.parse_args(
        [
            "advisor",
            "-i",
            f"{os.getcwd()}",
            "-s",
            f"{os.getcwd()}",
            "-t",
            "ttft",
            "-m",
            "average",
            "-in",
            "100",
            "-out",
            "50",
            "-tp",
            "2",
            "-l",
            "debug",
        ]
    )

    assert args.instance_path == f"{os.getcwd()}"
    assert args.service_config_path == f"{os.getcwd()}"
    assert args.target == "ttft"
    assert args.target_metrics == "average"
    assert args.input_token_num == 100
    assert args.output_token_num == 50
    assert args.tp == 2
    assert args.log_level == "debug"
    assert args.func == advisor.main


def test_arg_parse_default_values():
    # Create actual parser and subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Add our advisor subparser
    advisor.arg_parse(subparsers)

    # Test with minimal arguments (using defaults)
    args = parser.parse_args(["advisor", "-i", f"{os.getcwd()}", "-s", f"{os.getcwd()}"])

    assert args.instance_path == f"{os.getcwd()}"
    assert args.service_config_path == f"{os.getcwd()}"
    assert args.target == "ttft"
    assert args.target_metrics == "average"
    assert args.input_token_num == 0
    assert args.output_token_num == 0
    assert args.tp == 0
    assert args.log_level == "info"


def test_arg_parse_with_environment_variable():
    # Set environment variable
    test_path = f"{os.getcwd()}"
    os.environ[advisor.MIES_INSTALL_PATH] = test_path

    try:
        # Create actual parser and subparsers
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        # Add our advisor subparser
        advisor.arg_parse(subparsers)

        # Test with minimal arguments
        args = parser.parse_args(["advisor"])
        service_config_path = advisor.get_mindie_server_config_path(args.service_config_path)

        assert service_config_path == os.path.join(test_path, "conf", "config.json")
    finally:
        # Clean up environment
        if advisor.MIES_INSTALL_PATH in os.environ:
            del os.environ[advisor.MIES_INSTALL_PATH]


def test_arg_parse_target_choices():
    os.environ[advisor.MIES_INSTALL_PATH] = f"{os.getcwd()}"
    # Create actual parser and subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Add our advisor subparser
    advisor.arg_parse(subparsers)

    # Test all valid target choices
    for target in advisor.TARGETS_MAP.keys():
        args = parser.parse_args(["advisor", "-i", f"{os.getcwd()}", "-t", target])
        assert args.target == target


def test_arg_parse_target_metrics_choices():
    os.environ[advisor.MIES_INSTALL_PATH] = f"{os.getcwd()}"
    # Create actual parser and subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Add our advisor subparser
    advisor.arg_parse(subparsers)

    # Test all valid metrics choices
    for metric in advisor.PERF_METRICS:
        args = parser.parse_args(["advisor", "-i", f"{os.getcwd()}", "-m", metric])
        assert args.target_metrics == metric


def test_arg_parse_invalid_positive_integer():
    # Create actual parser and subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Add our advisor subparser
    advisor.arg_parse(subparsers)

    # Test invalid integer values
    with pytest.raises(SystemExit):
        parser.parse_args(["advisor", "-i", "test_instance", "-in", "-1"])  # Negative number

    with pytest.raises(SystemExit):
        parser.parse_args(["advisor", "-i", "test_instance", "-tp", "abc"])  # Not a number
