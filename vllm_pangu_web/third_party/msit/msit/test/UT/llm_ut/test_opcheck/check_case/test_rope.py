import sys
from unittest.mock import patch, MagicMock
import pytest
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_rope_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.rope import OpcheckUnpadRopeOperation
    OpcheckUnpadRopeOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckUnpadRopeOperation": OpcheckUnpadRopeOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_rotate_half_given_valid_input_when_even_dim_then_correct_output(import_rope_module):
    OpcheckUnpadRopeOperation = import_rope_module["OpcheckUnpadRopeOperation"]
    op = OpcheckUnpadRopeOperation()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = op.rotate_half(x)

    expected = torch.tensor([[-2.0, 1.0], [-4.0, 3.0]])
    assert torch.equal(result, expected)


def test_golden_func1_given_valid_input_when_batch_3_then_correct_output(import_rope_module):
    OpcheckUnpadRopeOperation = import_rope_module["OpcheckUnpadRopeOperation"]
    op = OpcheckUnpadRopeOperation()
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4, 4, 4])  # seqlen
    ]

    result = op.golden_func1(in_tensors)

    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)


def test_golden_func2_given_valid_input_when_batch_3_then_correct_output(import_rope_module):
    OpcheckUnpadRopeOperation = import_rope_module["OpcheckUnpadRopeOperation"]
    op = OpcheckUnpadRopeOperation()
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4])  # seqlen
    ]

    result = op.golden_func2(in_tensors)

    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)


def test_golden_func3_given_valid_input_when_batch_3_then_correct_output(import_rope_module):
    OpcheckUnpadRopeOperation = import_rope_module["OpcheckUnpadRopeOperation"]
    op = OpcheckUnpadRopeOperation()
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4])  # seqlen
    ]

    result = op.golden_func3(in_tensors)

    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)


def test_golden_func4_given_valid_input_when_batch_3_then_correct_output(import_rope_module):
    OpcheckUnpadRopeOperation = import_rope_module["OpcheckUnpadRopeOperation"]
    op = OpcheckUnpadRopeOperation()
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4])  # seqlen
    ]

    result = op.golden_func4(in_tensors)

    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)


def test_golden_calc_given_valid_input_when_rotaryCoeff_4_then_correct_output(import_rope_module):
    OpcheckUnpadRopeOperation = import_rope_module["OpcheckUnpadRopeOperation"]
    op = OpcheckUnpadRopeOperation()
    op.op_param = {'rotaryCoeff': 4}
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4, 4, 4])  # seqlen
    ]

    result = op.golden_calc(in_tensors)

    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)


def test_golden_calc_given_valid_input_when_rotaryCoeff_64_then_correct_output(import_rope_module):
    OpcheckUnpadRopeOperation = import_rope_module["OpcheckUnpadRopeOperation"]
    op = OpcheckUnpadRopeOperation()
    op.op_param = {'rotaryCoeff': 64}
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4])  # seqlen
    ]

    result = op.golden_calc(in_tensors)

    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)


def test_golden_calc_given_valid_input_when_rotaryCoeff_none_then_correct_output(import_rope_module):
    OpcheckUnpadRopeOperation = import_rope_module["OpcheckUnpadRopeOperation"]
    op = OpcheckUnpadRopeOperation()
    op.op_param = {}
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4])  # seqlen
    ]

    result = op.golden_calc(in_tensors)

    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)


def test_test_given_invalid_params_when_missing_rotaryCoeff_then_return_early(import_rope_module):
    OpcheckUnpadRopeOperation = import_rope_module["OpcheckUnpadRopeOperation"]
    op = OpcheckUnpadRopeOperation()
    op.op_param = {}

    result = op.test()

    assert result is None


def test_validate_param_given_valid_params_when_rotaryCoeff_4_then_return_true(import_rope_module):
    OpcheckUnpadRopeOperation = import_rope_module["OpcheckUnpadRopeOperation"]
    op = OpcheckUnpadRopeOperation()
    op.op_param = {'rotaryCoeff': 4}

    result = op.validate_param("rotaryCoeff")

    assert result is True


def test_validate_param_given_invalid_params_when_missing_rotaryCoeff_then_return_false(import_rope_module):
    OpcheckUnpadRopeOperation = import_rope_module["OpcheckUnpadRopeOperation"]
    op = OpcheckUnpadRopeOperation()
    op.op_param = {}

    result = op.validate_param("rotaryCoeff")

    assert result is False
