import sys
from unittest.mock import patch, MagicMock
import pytest
import torch


from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_onehot_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.onehot import OpcheckOnehotOperation
    OpcheckOnehotOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckOnehotOperation": OpcheckOnehotOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_valid_input_when_axis_0_depth_3_then_correct_output(import_onehot_module):
    OpcheckOnehotOperation = import_onehot_module["OpcheckOnehotOperation"]
    op = OpcheckOnehotOperation()
    op.op_param = {'axis': 0, 'depth': 3}
    in_tensors = torch.tensor([0, 1, 2])

    result = op.golden_calc([in_tensors])

    expected = torch.eye(3)[in_tensors]
    assert torch.equal(result[0], expected)


def test_golden_calc_given_invalid_axis_when_axis_out_of_range_then_raise_error(import_onehot_module):
    OpcheckOnehotOperation = import_onehot_module["OpcheckOnehotOperation"]
    op = OpcheckOnehotOperation()
    op.op_param = {'axis': 2, 'depth': 3}
    in_tensors = torch.tensor([0, 1, 2])

    with pytest.raises(IndexError):
        op.golden_calc([in_tensors])


def test_test_given_invalid_params_when_missing_depth_then_return_early(import_onehot_module):
    OpcheckOnehotOperation = import_onehot_module["OpcheckOnehotOperation"]
    op = OpcheckOnehotOperation()
    op.op_param = {'axis': 0}

    result = op.test()

    assert result is None


def test_test_given_invalid_params_when_missing_axis_then_return_early(import_onehot_module):
    OpcheckOnehotOperation = import_onehot_module["OpcheckOnehotOperation"]
    op = OpcheckOnehotOperation()
    op.op_param = {'depth': 3}

    result = op.test()

    assert result is None


def test_test_given_invalid_params_when_missing_both_axis_and_depth_then_return_early(import_onehot_module):
    OpcheckOnehotOperation = import_onehot_module["OpcheckOnehotOperation"]
    op = OpcheckOnehotOperation()
    op.op_param = {}

    result = op.test()

    assert result is None


def test_validate_param_given_valid_params_when_axis_0_depth_3_then_return_true(import_onehot_module):
    OpcheckOnehotOperation = import_onehot_module["OpcheckOnehotOperation"]
    op = OpcheckOnehotOperation()
    op.op_param = {'axis': 0, 'depth': 3}

    result = op.validate_param("axis", "depth")

    assert result is True


def test_validate_param_given_invalid_params_when_missing_depth_then_return_false(import_onehot_module):
    OpcheckOnehotOperation = import_onehot_module["OpcheckOnehotOperation"]
    op = OpcheckOnehotOperation()
    op.op_param = {'axis': 0}

    result = op.validate_param("axis", "depth")

    assert result is False


def test_validate_param_given_invalid_params_when_missing_axis_then_return_false(import_onehot_module):
    OpcheckOnehotOperation = import_onehot_module["OpcheckOnehotOperation"]
    op = OpcheckOnehotOperation()
    op.op_param = {'depth': 3}

    result = op.validate_param("axis", "depth")

    assert result is False


def test_validate_param_given_invalid_params_when_missing_both_axis_and_depth_then_return_false(import_onehot_module):
    OpcheckOnehotOperation = import_onehot_module["OpcheckOnehotOperation"]
    op = OpcheckOnehotOperation()
    op.op_param = {}

    result = op.validate_param("axis", "depth")

    assert result is False
