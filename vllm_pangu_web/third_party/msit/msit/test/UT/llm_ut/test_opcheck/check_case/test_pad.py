import sys
import pytest
from unittest.mock import MagicMock
import torch


from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_pad_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.pad import OpcheckPadOperation
    OpcheckPadOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckPadOperation": OpcheckPadOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_valid_input_when_batch_1_then_correct_output(import_pad_module):
    OpcheckPadOperation = import_pad_module["OpcheckPadOperation"]
    op = OpcheckPadOperation()
    in_tensors = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # tmp_out
        torch.tensor([0]),  # padding_offset
        torch.tensor([[2]]),  # seq_len
        torch.tensor([[1, 2]])  # input_ids
    ]

    result = op.golden_calc(in_tensors)

    expected = torch.tensor([[3.0, 4.0]])
    assert torch.equal(result[0], expected)


def test_golden_calc_given_invalid_input_when_seq_len_mismatch_then_raise_error(import_pad_module):
    OpcheckPadOperation = import_pad_module["OpcheckPadOperation"]
    op = OpcheckPadOperation()
    in_tensors = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # tmp_out
        torch.tensor([0]),  # padding_offset
        torch.tensor([[3]]),  # seq_len
        torch.tensor([[1, 2]])  # input_ids
    ]

    with pytest.raises(IndexError):
        op.golden_calc(in_tensors)
