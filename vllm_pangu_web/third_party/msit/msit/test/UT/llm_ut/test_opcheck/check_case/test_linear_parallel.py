import sys
from unittest.mock import patch, MagicMock

import pytest
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_linear_parallel_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.linear_parallel import OpcheckLinearParallelOperation, ParallelType
    OpcheckLinearParallelOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckLinearParallelOperation": OpcheckLinearParallelOperation,
        "ParallelType": ParallelType
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_test_when_valid_input(import_linear_parallel_module):
    ParallelType = import_linear_parallel_module['ParallelType']
    test_cases = [
        ({'backend': 'hccl', 'rank': 0, 'rankSize': 2}, True),
        ({'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.LINEAR_ALL_REDUCE.value}, True),
        ({'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.LINEAR_REDUCE_SCATTER.value}, True),
        ({'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.ALL_GATHER_LINEAR.value}, True),
        ({'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.PURE_LINEAR.value}, True),
    ]
    for op_param, expected_execute_call in test_cases:
        OpcheckLinearParallelOperation = import_linear_parallel_module['OpcheckLinearParallelOperation']
        op = OpcheckLinearParallelOperation()
        op.op_param = op_param
        with patch.object(op, 'get_soc_version', return_value='Ascend910B'):
            with patch.object(op, 'validate_param', return_value=True):
                with patch.object(op, 'execute') as mock_execute:
                    op.test()
                    if expected_execute_call:
                        mock_execute.assert_called_once()
                    else:
                        mock_execute.assert_not_called()
