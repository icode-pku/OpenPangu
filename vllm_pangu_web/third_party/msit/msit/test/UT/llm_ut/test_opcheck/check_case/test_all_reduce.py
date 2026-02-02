import sys
from unittest.mock import patch, MagicMock

import torch
import pytest
    
from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_allreduce_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case import OpcheckAllReduceOperation
    OpcheckAllReduceOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckAllReduceOperation": OpcheckAllReduceOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_sum_cal_given_in_tensors_when_valid_input_then_correct_result(import_allreduce_module):
    test_cases = [
        ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([4.0, 6.0])]),
        ([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])], [torch.tensor([6.0])]),
        ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])], [torch.tensor([9.0, 12.0])]),
    ]
    for in_tensors, expected_result in test_cases:
        OpcheckAllReduceOperation = import_allreduce_module["OpcheckAllReduceOperation"]
        result = OpcheckAllReduceOperation.sum_cal(in_tensors)

        assert torch.equal(result[0], expected_result[0])


def test_max_cal_given_in_tensors_when_valid_input_then_correct_result(import_allreduce_module):
    test_cases = [
        ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([3.0, 4.0])]),
        ([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])], [torch.tensor([3.0])]),
        ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])], [torch.tensor([5.0, 6.0])]),
    ]
    for in_tensors, expected_result in test_cases:
        OpcheckAllReduceOperation = import_allreduce_module["OpcheckAllReduceOperation"]
        result = OpcheckAllReduceOperation.max_cal(in_tensors)

        assert torch.equal(result[0], expected_result[0])


def test_min_cal_given_in_tensors_when_valid_input_then_correct_result(import_allreduce_module):
    test_cases = [
        ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([1.0, 2.0])]),
        ([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])], [torch.tensor([1.0])]),
        ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])], [torch.tensor([1.0, 2.0])]),
    ]
    for in_tensors, expected_result in test_cases:
        OpcheckAllReduceOperation = import_allreduce_module["OpcheckAllReduceOperation"]
        result = OpcheckAllReduceOperation.min_cal(in_tensors)

        assert torch.equal(result[0], expected_result[0])


def test_prod_cal_given_in_tensors_when_valid_input_then_correct_result(import_allreduce_module):
    test_cases = [
        ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([3.0, 8.0])]),
        ([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])], [torch.tensor([6.0])]),
        ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])], [torch.tensor([15.0, 48.0])]),
    ]
    for in_tensors, expected_result in test_cases:
        OpcheckAllReduceOperation = import_allreduce_module["OpcheckAllReduceOperation"]
        result = OpcheckAllReduceOperation.prod_cal(in_tensors)

        assert torch.equal(result[0], expected_result[0])


def test_golden_calc_given_all_reduce_type_in_tensors_when_valid_input_then_correct_result(import_allreduce_module):
    test_cases = [
        ("sum", [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([4.0, 6.0])]),
        ("max", [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([3.0, 4.0])]),
        ("min", [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([1.0, 2.0])]),
        ("prod", [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([3.0, 8.0])]),
    ]
    for all_reduce_type, in_tensors, expected_result in test_cases:
        OpcheckAllReduceOperation = import_allreduce_module["OpcheckAllReduceOperation"]
        
        op = OpcheckAllReduceOperation()
        op.op_param = {'allReduceType': all_reduce_type}

        def get_new_in_tensors():
            return in_tensors

        op.get_new_in_tensors = get_new_in_tensors

        result = op.golden_calc(None)

        assert torch.equal(result[0], expected_result[0])


def test_test_all_reduce_given_pid_op_param_when_valid_input_then_execute_successfully(import_allreduce_module):
    test_cases = [
        (1, {'allReduceType': 'sum', 'rank': 0, 'rankRoot': 0, 'rankSize': 2}, True, True),
        (None, {'allReduceType': 'sum', 'rank': 0, 'rankRoot': 0, 'rankSize': 2}, True, False),
        (1, {'allReduceType': 'sum', 'rank': 0, 'rankRoot': 0, 'rankSize': 2}, False, False),
        (1, {'rank': 0, 'rankRoot': 0, 'rankSize': 2}, False, False),
        (1, {'allReduceType': 'sum', 'rankRoot': 0, 'rankSize': 2}, False, False),
        (1, {'allReduceType': 'sum', 'rank': 0, 'rankSize': 2}, False, False),
        (1, {'allReduceType': 'sum', 'rank': 0, 'rankRoot': 0}, False, False),
    ]
    for pid, op_param, validate_param_return, expected_execute_call in test_cases:
        OpcheckAllReduceOperation = import_allreduce_module["OpcheckAllReduceOperation"]
        op = OpcheckAllReduceOperation()
        op.pid = pid
        op.op_param = op_param

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test_all_reduce()


        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()
