import sys
from unittest.mock import patch, MagicMock
import pytest
import torch
from mock_operation_test import MockOperationTest

@pytest.fixture(scope="function")
def import_index_add_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    
    from msit_llm.opcheck.check_case import OpcheckIndexAddOperation
    OpcheckIndexAddOperation.__bases__ = (MockOperationTest,)
    
    yield {"OpcheckIndexAddOperation": OpcheckIndexAddOperation}
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]

def test_golden_calc_when_valid_input(import_index_add_module):
    test_cases = [
        {
            "op_param": {'indexType': 1, 'axis': 0},
            "in_tensors": [torch.randn(2, 4, 3, 3), torch.tensor([0,1]), torch.randn(2, 4, 3, 3), torch.tensor(1.0)],
            "expected_result": True
        },
        {
            "op_param": {'indexType': 0, 'axis': 0},
            "in_tensors": [torch.randn(2, 4, 3, 3), torch.tensor([0,1]), torch.randn(2, 4, 3, 3), torch.tensor(1.0)],
            "expected_result": False
        }
    ]
    
    for case in test_cases:
        OpcheckIndexAddOperation = import_index_add_module['OpcheckIndexAddOperation']
        op = OpcheckIndexAddOperation()
        op.op_param = case["op_param"]
        result = op.golden_calc(case["in_tensors"])
        
        if case["expected_result"]:
            assert result
            assert len(result) > 0
        else:
            assert not result

def test_test_when_valid_input(import_index_add_module):
    test_cases = [
        ({'indexType': 1, 'axis': 0}, True, True),
        ({'indexType': 1, 'axis': 1}, True, True),
        ({'indexType': 0, 'axis': 0}, True, True),
        ({'indexType': 0, 'axis': 1}, True, True),
        ({'indexType': 1}, False, False),
        ({'axis': 0}, False, False),
        ({}, False, False),
    ]
    
    for op_param, validate_param_return, expected_execute_call in test_cases:
        OpcheckIndexAddOperation = import_index_add_module['OpcheckIndexAddOperation']
        op = OpcheckIndexAddOperation()
        op.op_param = op_param

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test()

        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()