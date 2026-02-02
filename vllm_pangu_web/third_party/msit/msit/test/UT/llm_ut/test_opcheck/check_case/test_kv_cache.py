import sys
from unittest.mock import patch, MagicMock

import pytest
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_kv_cache_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case import OpcheckKvCacheOperation
    OpcheckKvCacheOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckKvCacheOperation": OpcheckKvCacheOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_test_when_valid_input(import_kv_cache_module):
    OpcheckKvCacheOperation = import_kv_cache_module["OpcheckKvCacheOperation"]
    op = OpcheckKvCacheOperation()
    op.op_param = {}

    with patch.object(op, 'execute_inplace') as mock_execute:
        op.test()

    mock_execute.assert_called_once()
