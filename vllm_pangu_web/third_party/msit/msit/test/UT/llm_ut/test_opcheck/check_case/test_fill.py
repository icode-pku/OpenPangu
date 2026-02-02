import sys
from unittest.mock import patch, MagicMock

import pytest
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_fill_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    
    from msit_llm.opcheck.check_case import OpcheckFillOperation
    OpcheckFillOperation.__bases__ = (MockOperationTest,)
    
    yield {
        "OpcheckFillOperation": OpcheckFillOperation
    }
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_op_param_in_tensors_when_valid_input_then_correct_result(import_fill_module):
    OpcheckFillOperation = import_fill_module['OpcheckFillOperation']
    
    test_cases = [
        {
            "op_param": {'withMask': True, 'value': [1.0]},
            "in_tensors": [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([True, False, True])],
            "expected_result": [torch.tensor([1.0, 2.0, 1.0])]
        },
        {
            "op_param": {'withMask': True, 'value': [0.0]},
            "in_tensors": [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([False, True, False])],
            "expected_result": [torch.tensor([1.0, 0.0, 3.0])]
        }
    ]
    
    for case in test_cases:
        op = OpcheckFillOperation()
        op.op_param = case["op_param"]
        
        result = op.golden_calc(case["in_tensors"])
        assert torch.allclose(result[0], case["expected_result"][0], atol=1e-4), \
            f"Failed for op_param={case['op_param']}"


def test_golden_calc_given_op_param_in_tensors_when_invalid_input_then_raise_error(import_fill_module):
    OpcheckFillOperation = import_fill_module['OpcheckFillOperation']
    
    test_cases = [
        {
            "op_param": {'withMask': True, 'value': [1.0]},
            "in_tensors": [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([True, False])],
            "expected_error": RuntimeError
        }
    ]
    
    for case in test_cases:
        op = OpcheckFillOperation()
        op.op_param = case["op_param"]
        
        with pytest.raises(case["expected_error"]):
            op.golden_calc(case["in_tensors"])


def test_test_given_op_param_when_valid_input_then_execute_successfully(import_fill_module):
    OpcheckFillOperation = import_fill_module['OpcheckFillOperation']
    
    test_cases = [
        ({'withMask': True, 'value': [1.0]}, True, True),
        ({'withMask': False, 'outDim': [2, 2], 'value': [0.0]}, True, True),
        ({'withMask': True, 'value': [0.0]}, True, True),
        ({'withMask': False, 'outDim': [3, 3], 'value': [1.0]}, True, True),
        ({'withMask': True}, False, False),
        ({'withMask': False, 'outDim': [2, 2]}, False, False),
        ({'withMask': True, 'value': [1.0]}, False, False),
        ({'withMask': False, 'outDim': [2, 2], 'value': []}, False, False),
    ]
    
    for op_param, validate_param_return, expected_execute_call in test_cases:
        op = OpcheckFillOperation()
        op.op_param = op_param

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test()

        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()