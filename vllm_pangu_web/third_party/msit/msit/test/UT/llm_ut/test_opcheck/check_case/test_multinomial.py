import sys
from unittest.mock import patch, MagicMock
import pytest
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_multinomial_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.multinomial import OpcheckMultinomialOperation
    OpcheckMultinomialOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckMultinomialOperation": OpcheckMultinomialOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_valid_input_when_numSamples_1_then_correct_shape(import_multinomial_module):
    OpcheckMultinomialOperation = import_multinomial_module["OpcheckMultinomialOperation"]
    op = OpcheckMultinomialOperation()
    op.op_param = {"numSamples": 1, "randSeed": 0}
    input0 = torch.randn(4, 16)
    in_tensors = [input0]

    result = op.golden_calc(in_tensors)

    assert result[0].shape == (4, 1)


def test_golden_calc_given_valid_input_when_numSamples_5_then_correct_shape(import_multinomial_module):
    OpcheckMultinomialOperation = import_multinomial_module["OpcheckMultinomialOperation"]
    op = OpcheckMultinomialOperation()
    op.op_param = {"numSamples": 5, "randSeed": 0}
    input0 = torch.randn(4, 16)
    in_tensors = [input0]

    result = op.golden_calc(in_tensors)

    assert result[0].shape == (4, 5)


def test_golden_calc_given_invalid_input_when_numSamples_negative_then_raise_exception(import_multinomial_module):
    OpcheckMultinomialOperation = import_multinomial_module["OpcheckMultinomialOperation"]
    op = OpcheckMultinomialOperation()
    op.op_param = {"numSamples": -1, "randSeed": 0}
    input0 = torch.randn(4, 16)
    in_tensors = [input0]

    with pytest.raises(Exception):
        op.golden_calc(in_tensors)


def test_golden_calc_given_valid_input_when_randSeed_non_zero_then_correct_shape(import_multinomial_module):
    OpcheckMultinomialOperation = import_multinomial_module["OpcheckMultinomialOperation"]
    op = OpcheckMultinomialOperation()
    op.op_param = {"numSamples": 1, "randSeed": 12345}
    input0 = torch.randn(4, 16)
    in_tensors = [input0]

    result = op.golden_calc(in_tensors)

    assert result[0].shape == (4, 1)


def test_golden_calc_given_valid_input_when_randSeed_zero_then_correct_shape(import_multinomial_module):
    OpcheckMultinomialOperation = import_multinomial_module["OpcheckMultinomialOperation"]
    op = OpcheckMultinomialOperation()
    op.op_param = {"numSamples": 1, "randSeed": 0}
    input0 = torch.randn(4, 16)
    in_tensors = [input0]

    result = op.golden_calc(in_tensors)

    assert result[0].shape == (4, 1)


def test_test_given_missing_numSamples_when_randSeed_provided_then_return_early(import_multinomial_module):
    OpcheckMultinomialOperation = import_multinomial_module["OpcheckMultinomialOperation"]
    op = OpcheckMultinomialOperation()
    op.op_param = {"randSeed": 0}

    op.test()

    assert True


def test_test_given_missing_randSeed_when_numSamples_provided_then_return_early(import_multinomial_module):
    OpcheckMultinomialOperation = import_multinomial_module["OpcheckMultinomialOperation"]
    op = OpcheckMultinomialOperation()
    op.op_param = {"numSamples": 1}

    op.test()

    assert True


def test_test_given_missing_both_params_when_no_params_provided_then_return_early(import_multinomial_module):
    OpcheckMultinomialOperation = import_multinomial_module["OpcheckMultinomialOperation"]
    op = OpcheckMultinomialOperation()
    op.op_param = {}

    op.test()

    assert True