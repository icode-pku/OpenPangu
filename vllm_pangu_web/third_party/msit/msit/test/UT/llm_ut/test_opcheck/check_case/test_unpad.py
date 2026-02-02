import sys
from unittest.mock import MagicMock

import pytest
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_unpad_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case import OpcheckUnpadOperation
    OpcheckUnpadOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckUnpadOperation": OpcheckUnpadOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_invalid_input_when_seq_len_mismatch_then_raise_error(import_unpad_module):
    OpcheckUnpadOperation = import_unpad_module["OpcheckUnpadOperation"]
    op = OpcheckUnpadOperation()
    in_tensors = [
        torch.tensor([[1, 2, 3], [4, 5, 6]]),  # input_ids
        torch.tensor([0]),  # cum_offsets_now
        torch.tensor([[2]]),  # token_num
        torch.tensor([[3]])  # seq_len
    ]

    with pytest.raises(IndexError):
        op.golden_calc(in_tensors)


def test_golden_calc_given_invalid_input_when_token_num_mismatch_then_raise_error(import_unpad_module):
    OpcheckUnpadOperation = import_unpad_module["OpcheckUnpadOperation"]
    op = OpcheckUnpadOperation()
    in_tensors = [
        torch.tensor([[1, 2, 3], [4, 5, 6]]),  # input_ids
        torch.tensor([0]),  # cum_offsets_now
        torch.tensor([[5]]),  # token_num
        torch.tensor([[3]])  # seq_len
    ]

    with pytest.raises(IndexError):
        op.golden_calc(in_tensors)


def test_golden_calc_given_invalid_input_when_seq_len_exceeds_then_raise_error(import_unpad_module):
    OpcheckUnpadOperation = import_unpad_module["OpcheckUnpadOperation"]
    op = OpcheckUnpadOperation()
    in_tensors = [
        torch.tensor([[1, 2, 3], [4, 5, 6]]),  # input_ids
        torch.tensor([0]),  # cum_offsets_now
        torch.tensor([[3]]),  # token_num
        torch.tensor([[4]])   # seq_len exceeds input_ids
    ]

    with pytest.raises(IndexError):
        op.golden_calc(in_tensors)


def test_test_when_execute_called_then_no_exception_raised(import_unpad_module):
    OpcheckUnpadOperation = import_unpad_module["OpcheckUnpadOperation"]
    op = OpcheckUnpadOperation()
    op.execute = MagicMock()  # Mock the execute method to do nothing
    
    try:
        op.test()
    except Exception as e:
        pytest.fail(f"test() raised an exception: {e}")
