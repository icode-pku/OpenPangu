import sys
from unittest.mock import patch, MagicMock
import pytest
import torch

from mock_operation_test import MockOperationTest

@pytest.fixture(scope="function")
def import_gating_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    
    from msit_llm.opcheck.check_case import OpcheckGatingOperation
    OpcheckGatingOperation.__bases__ = (MockOperationTest,)
    
    yield {"OpcheckGatingOperation": OpcheckGatingOperation}
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]

def test_test_when_valid_input(import_gating_module):
    test_cases = [
        ({'topkExpertNum': 2, 'cumSumNum': 3}, True, True),
        ({'topkExpertNum': 1, 'cumSumNum': 2}, True, True),
        ({'topkExpertNum': 3, 'cumSumNum': 4}, True, True),
        ({'topkExpertNum': 2}, False, False),
        ({'cumSumNum': 3}, False, False),
        ({}, False, False),
    ]
    
    for op_param, validate_param_return, expected_execute_call in test_cases:
        OpcheckGatingOperation = import_gating_module['OpcheckGatingOperation']
        op = OpcheckGatingOperation()
        op.op_param = op_param

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test()

        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()