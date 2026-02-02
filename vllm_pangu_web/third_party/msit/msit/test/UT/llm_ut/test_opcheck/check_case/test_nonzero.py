import sys
import pytest
from unittest.mock import patch, MagicMock
import torch


from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_nonzero_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.nonzero import OpcheckNonzeroOperation
    OpcheckNonzeroOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckNonzeroOperation": OpcheckNonzeroOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_positive_tensor_when_valid_input_then_correct_result(import_nonzero_module):
    OpcheckNonzeroOperation = import_nonzero_module["OpcheckNonzeroOperation"]
    in_tensors = [torch.tensor([[1, 2], [3, 4]])]
    op = OpcheckNonzeroOperation()

    result = op.golden_calc(in_tensors)

    assert result[0].shape == (2, 4)
    assert result[1] == torch.tensor(4).long()


def test_golden_calc_given_mixed_tensor_when_valid_input_then_correct_result(import_nonzero_module):
    OpcheckNonzeroOperation = import_nonzero_module["OpcheckNonzeroOperation"]
    in_tensors = [torch.tensor([[1, -2], [0, 4]])]
    op = OpcheckNonzeroOperation()

    result = op.golden_calc(in_tensors)

    assert result[0].shape == (2, 4)
    assert result[1] == torch.tensor(3).long()

def test_golden_calc_given_all_zero_tensor_when_valid_input_then_correct_result(import_nonzero_module):
    OpcheckNonzeroOperation = import_nonzero_module["OpcheckNonzeroOperation"]
    in_tensors = [torch.tensor([[0, 0], [0, 0]])]
    op = OpcheckNonzeroOperation()

    result = op.golden_calc(in_tensors)

    assert result[0].shape == (2, 4)
    assert result[1] == torch.tensor(0).long()


def test_golden_calc_given_empty_tensor_when_valid_input_then_correct_result(import_nonzero_module):
    OpcheckNonzeroOperation = import_nonzero_module["OpcheckNonzeroOperation"]
    in_tensors = [torch.tensor([])]
    op = OpcheckNonzeroOperation()

    result = op.golden_calc(in_tensors)

    assert result[0].shape == (1, 0)
    assert result[1] == torch.tensor(0).long()


def test_golden_calc_given_tensor_with_nan_when_valid_input_then_correct_result(import_nonzero_module):
    OpcheckNonzeroOperation = import_nonzero_module["OpcheckNonzeroOperation"]
    in_tensors = [torch.tensor([[1, float('nan')], [float('nan'), 4]])]
    op = OpcheckNonzeroOperation()

    result = op.golden_calc(in_tensors)

    assert result[0].shape == (2, 4)
    assert result[1] == torch.tensor(4).long()


def test_golden_calc_given_tensor_with_inf_when_valid_input_then_correct_result(import_nonzero_module):
    OpcheckNonzeroOperation = import_nonzero_module["OpcheckNonzeroOperation"]
    in_tensors = [torch.tensor([[1, float('inf')], [float('inf'), 4]])]
    op = OpcheckNonzeroOperation()

    result = op.golden_calc(in_tensors)

    assert result[0].shape == (2, 4)
    assert result[1] == torch.tensor(4).long()


def test_golden_calc_given_tensor_with_negative_inf_when_valid_input_then_correct_result(import_nonzero_module):
    OpcheckNonzeroOperation = import_nonzero_module["OpcheckNonzeroOperation"]
    in_tensors = [torch.tensor([[1, float('-inf')], [float('-inf'), 4]])]
    op = OpcheckNonzeroOperation()

    result = op.golden_calc(in_tensors)

    assert result[0].shape == (2, 4)
    assert result[1] == torch.tensor(4).long()
