import sys
from unittest.mock import patch, MagicMock
import pytest
import torch

from mock_operation_test import MockOperationTest

@pytest.fixture(scope="function")
def import_elewise_sub_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    
    from msit_llm.opcheck.check_case import OpcheckElewiseSubOperation
    OpcheckElewiseSubOperation.__bases__ = (MockOperationTest,)
    
    yield {"OpcheckElewiseSubOperation": OpcheckElewiseSubOperation}
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]

def test_test_2d_half_when_valid_input(import_elewise_sub_module):
    test_cases = [
        ({'seqLen': [2, 3], 'headNum': 2}, True, True),
        ({'seqLen': [1, 2], 'headNum': 1}, True, True),
        ({'seqLen': [3, 3], 'headNum': 3}, True, True),
        ({'seqLen': [2, 3]}, False, False),
        ({'headNum': 2}, False, False),
        ({}, False, False),
    ]
    
    for op_param, validate_param_return, expected_execute_call in test_cases:
        OpcheckElewiseSubOperation = import_elewise_sub_module['OpcheckElewiseSubOperation']
        op = OpcheckElewiseSubOperation()
        op.op_param = op_param

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test_2d_half()

        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()