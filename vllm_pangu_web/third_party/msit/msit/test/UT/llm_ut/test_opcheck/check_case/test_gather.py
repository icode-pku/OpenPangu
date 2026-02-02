import sys
from unittest.mock import patch, MagicMock

import pytest
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_gather_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    
    from msit_llm.opcheck.check_case import OpcheckGatherOperation
    OpcheckGatherOperation.__bases__ = (MockOperationTest,)
    
    yield {
        "OpcheckGatherOperation": OpcheckGatherOperation
    }
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_test_when_valid_input(import_gather_module):
    OpcheckGatherOperation = import_gather_module['OpcheckGatherOperation']
    
    test_cases = [
        ({'axis': 0, 'batchDims': 0}, True, True),
        ({'axis': 1, 'batchDims': 0}, True, True),
        ({'axis': 0, 'batchDims': 1}, True, True),
        ({'axis': 1, 'batchDims': 1}, True, True),
        ({'axis': 0}, False, False),
        ({'batchDims': 0}, False, False),
        ({}, False, False),
    ]
    
    for op_param, validate_param_return, expected_execute_call in test_cases:
        op = OpcheckGatherOperation()
        op.op_param = op_param

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test()

        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()