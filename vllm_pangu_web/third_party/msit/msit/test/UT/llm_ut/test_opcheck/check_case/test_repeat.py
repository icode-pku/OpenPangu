import sys
import pytest
from unittest.mock import MagicMock
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_repeat_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.repeat import OpcheckRepeatOperation
    OpcheckRepeatOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckRepeatOperation": OpcheckRepeatOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_valid_multiples_when_1d_tensor_then_correct_result(import_repeat_module):
    OpcheckRepeatOperation = import_repeat_module["OpcheckRepeatOperation"]
    op = OpcheckRepeatOperation()
    op.op_param = {"multiples": (2,)}
    in_tensors = [torch.tensor([1, 2, 3])]

    result = op.golden_calc(in_tensors)

    expected = torch.tensor([1, 2, 3, 1, 2, 3])
    assert torch.allclose(result[0], expected)


def test_golden_calc_given_valid_multiples_when_2d_tensor_then_correct_result(import_repeat_module):
    OpcheckRepeatOperation = import_repeat_module["OpcheckRepeatOperation"]
    op = OpcheckRepeatOperation()
    op.op_param = {"multiples": (2, 3)}
    in_tensors = [torch.tensor([[1, 2], [3, 4]])]

    result = op.golden_calc(in_tensors)

    expected = torch.tensor([[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4], [1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]])
    assert torch.allclose(result[0], expected)


def test_golden_calc_given_valid_multiples_when_3d_tensor_then_correct_result(import_repeat_module):
    OpcheckRepeatOperation = import_repeat_module["OpcheckRepeatOperation"]
    op = OpcheckRepeatOperation()
    op.op_param = {"multiples": (2, 2, 2)}
    in_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]

    result = op.golden_calc(in_tensors)

    expected = torch.tensor([[[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]],
                             [[5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6], [7, 8, 7, 8]],
                             [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]],
                             [[5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6], [7, 8, 7, 8]]])
    assert torch.allclose(result[0], expected)


def test_golden_calc_given_invalid_multiples_when_3d_tensor_then_error(import_repeat_module):
    OpcheckRepeatOperation = import_repeat_module["OpcheckRepeatOperation"]
    op = OpcheckRepeatOperation()
    op.op_param = {"multiples": (2, 2)}
    in_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]

    with pytest.raises(RuntimeError):
        op.golden_calc(in_tensors)


def test_test_given_valid_multiples_when_execute_then_no_error(import_repeat_module):
    OpcheckRepeatOperation = import_repeat_module["OpcheckRepeatOperation"]
    op = OpcheckRepeatOperation()
    op.op_param = {"multiples": (2,)}

    def mock_validate_param(*args, **kwargs):
        True

    op.validate_param = mock_validate_param

    def mock_execute():
        pass

    op.execute = mock_execute

    op.test()

    assert True


def test_test_given_invalid_multiples_when_execute_then_return(import_repeat_module):
    OpcheckRepeatOperation = import_repeat_module["OpcheckRepeatOperation"]
    op = OpcheckRepeatOperation()
    op.op_param = {"multiples": None}

    def mock_validate_param(*args, **kwargs):
        False

    op.validate_param = mock_validate_param

    result = op.test()

    assert result is None
