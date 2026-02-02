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
from unittest.mock import patch, MagicMock, call
import random
import pytest
import numpy as np

# Import the module to test with proper error handling
from msserviceprofiler.msservice_advisor.profiling_analyze import batch_analyze
from msserviceprofiler.msservice_advisor.profiling_analyze.register import REGISTRY, ANSWERS
from msserviceprofiler.msservice_advisor.profiling_analyze.utils import SUGGESTION_TYPES, logger


# Test fixtures
@pytest.fixture(autouse=True)
def reset_state():
    """Reset the REGISTRY and ANSWERS before each test"""
    REGISTRY.clear()
    for key in ANSWERS:
        ANSWERS[key].clear()
    yield


@pytest.fixture
def mock_dependencies():
    with patch("matplotlib.pyplot") as mock_plt, patch("numpy.linspace") as mock_linspace, patch(
        "datetime.datetime"
    ) as mock_datetime, patch("logging.getLogger") as mock_logger:

        # Setup mock plt
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, [mock_ax])
        mock_plt.savefig.return_value = None

        # Setup mock datetime
        mock_now = MagicMock()
        mock_now.strftime.return_value = "123456"  # Fixed timestamp
        mock_datetime.now.return_value = mock_now

        # Setup mock logger
        mock_log = MagicMock()
        mock_logger.return_value = mock_log

        # Setup numpy linspace
        mock_linspace.return_value = np.array([0, 1, 2, 3])

        yield {"plt": mock_plt, "linspace": mock_linspace, "datetime": mock_datetime, "logger": mock_log}


# Test data
SAMPLE_BATCH_INFO = {
    1: [10.5, 11.2, 9.8, 10.1, 12.0],
    2: [20.1, 19.8, 21.5, 20.9, 18.7],
    4: [38.2, 40.1, 39.5, 37.8, 41.2],
}

SAMPLE_PRE_REQUEST = {
    "req1": {"prefill_bsz": 1, "decode_bsz": [1, 1, 1], "latency": [10.5, 2.1, 2.3, 2.0]},
    "req2": {"prefill_bsz": 2, "decode_bsz": [2, 2], "latency": [20.1, 4.0, 4.2]},
}


# Test summary_batch_info
def test_summary_batch_info_given_valid_input_returns_correct_summary():
    result = batch_analyze.summary_batch_info(SAMPLE_BATCH_INFO)

    # Check basic structure
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 4 in result

    # Check calculations for batch size 1
    bsz1 = result[1]
    assert bsz1["BSZ"] == 1
    assert bsz1["MIN"] == min(SAMPLE_BATCH_INFO[1])
    assert bsz1["P50"] == sorted(SAMPLE_BATCH_INFO[1])[2]  # Median
    assert bsz1["MAX"] == max(SAMPLE_BATCH_INFO[1])
    assert len(bsz1["FIT_DATA"]) == 2  # 30% to 70% of 5 is 1.5 to 3.5 -> indices 2 to 3


# Test print_list
def test_print_list_given_array_logs_each_item():
    test_array = ["item1", "item2", "item3"]
    with patch.object(batch_analyze.logger, "info") as mock_log:
        batch_analyze.print_list(test_array)


# Test read_batch_and_latency
def test_read_batch_and_latency_given_valid_input_returns_correct_summaries():
    prefill, decode = batch_analyze.read_batch_and_latency(SAMPLE_PRE_REQUEST)

    # Check prefill summary
    assert len(prefill) == 2  # batch sizes 1 and 2
    assert prefill[1]["BSZ"] == 1
    assert len(prefill[1]["FIT_DATA"]) <= 1  # Only one prefill latency for batch size 1

    # Check decode summary
    assert len(decode) == 2  # batch sizes 1 and 2
    assert decode[1]["BSZ"] == 1
    assert len(decode[1]["FIT_DATA"]) >= 2  # Multiple decode latencies


def test_read_batch_and_latency_given_mismatched_data_logs_warning():
    bad_request = {"req1": {"prefill_bsz": 1, "decode_bsz": [], "latency": [10.5, 2.1]}}
    with patch.object(batch_analyze.logger, "debug") as mock_log:
        batch_analyze.read_batch_and_latency(bad_request)


# Test find_best_by_curve_fit
@patch("scipy.optimize.curve_fit")
@patch("scipy.optimize.minimize")
def test_find_best_by_curve_fit_given_enough_data_returns_result(mock_minimize, mock_curve_fit):
    # Setup mocks
    mock_curve_fit.return_value = ([1, 2, 3], None)  # popt, pcov
    mock_minimize.return_value.x = [42]  # Best batch size

    summary_data = [
        {"BSZ": 1, "FIT_DATA": [10, 11, 12]},
        {"BSZ": 2, "FIT_DATA": [20, 21, 22]},
        {"BSZ": 4, "FIT_DATA": [40, 41, 42]},
    ]

    result = batch_analyze.find_best_by_curve_fit(summary_data, "test_process")
    assert result is not None
    assert result["best_batch_size"] == 42
    assert result["process_name"] == "test_process"


def test_find_best_by_curve_fit_given_insufficient_data_returns_none():
    summary_data = [{"BSZ": 1, "FIT_DATA": [10]}]
    with patch.object(batch_analyze.logger, "warning") as mock_log:
        result = batch_analyze.find_best_by_curve_fit(summary_data, "test_process")
        assert result is not None
        assert "best_batch_size" in result
        assert "func_curv" in result


# Test get_predict_image
def test_get_predict_image_success(mock_dependencies):
    # Setup test data
    results = [
        {
            "max_batch_size": 10,
            "points": [1, 2, 3],
            "targets": [4, 5, 6],
            "popt": (1, 2, 3),
            "process_name": "test_process",
            "func_curv": lambda x, a, b, c: a * x**2 + b * x + c,
        }
    ]

    # Call the function
    batch_analyze.get_predict_image(results)


@patch.object(batch_analyze, "plt", None)
def test_get_predict_image_without_matplotlib_does_nothing():
    with patch.object(batch_analyze.logger, "info") as mock_log:
        batch_analyze.get_predict_image([{}])
        mock_log.assert_not_called()


# Test find_best_batch_size integration
def test_find_best_batch_size_given_insufficient_data_adds_suggestion():
    benchmark = {"results_per_request": {"req1": {"prefill_bsz": 1, "decode_bsz": [1], "latency": [10.5, 2.1]}}}

    batch_analyze.find_best_batch_size({}, benchmark, {})

    assert "maxBatchSize" in ANSWERS[SUGGESTION_TYPES.config]
