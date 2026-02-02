import sys
import pytest
from unittest.mock import MagicMock
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_reduce_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.reduce import OpcheckReduceOperation, ReduceType
    OpcheckReduceOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckReduceOperation": OpcheckReduceOperation,
        "ReduceType": ReduceType
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_reduce_sum_and_no_axis_when_valid_input_then_correct_result(import_reduce_module):
    OpcheckReduceOperation = import_reduce_module["OpcheckReduceOperation"]
    ReduceType = import_reduce_module["ReduceType"]
    op_check = OpcheckReduceOperation()
    op_check.op_param = {'reduceType': ReduceType.REDUCE_SUM.value}
    in_tensors = [torch.tensor([[1, 2], [3, 4]])]

    result = op_check.golden_calc(in_tensors)

    assert result == [torch.tensor([10])]
