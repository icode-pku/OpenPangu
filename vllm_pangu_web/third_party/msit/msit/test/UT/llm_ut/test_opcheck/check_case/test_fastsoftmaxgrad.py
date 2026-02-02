import sys
from unittest.mock import patch, MagicMock

import pytest
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_fastsoftmaxgrad_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case import OpcheckFastSoftMaxGradOperation
    OpcheckFastSoftMaxGradOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckFastSoftMaxGradOperation": OpcheckFastSoftMaxGradOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


@pytest.mark.parametrize("batch_size_imm, head_num_imm, seq_len_range", [
    (4, 8, (100, 300)),
    (2, 4, (50, 150)),
    (1, 2, (25, 75)),
])
def test_test_fastsoftmaxgrad_given_batch_size_head_num_seq_len_range_when_valid_input_then_execute_successfully(
        batch_size_imm, head_num_imm, seq_len_range, import_fastsoftmaxgrad_module):
    OpcheckFastSoftMaxGradOperation = import_fastsoftmaxgrad_module['OpcheckFastSoftMaxGradOperation']
    op = OpcheckFastSoftMaxGradOperation()
    op.op_param = {}

    with patch.object(op, 'execute') as mock_execute:
        op.test_fastsoftmaxgrad()

    mock_execute.assert_called_once()


@pytest.mark.parametrize("batch_size_imm, head_num_imm, seq_len_range", [
    (4, 8, (100, 300)),
    (2, 4, (50, 150)),
    (1, 2, (25, 75)),
])
def test_test_fastsoftmaxgrad_given_batch_size_head_num_seq_len_range_when_valid_input_then_execute_successfully(
        batch_size_imm, head_num_imm, seq_len_range, import_fastsoftmaxgrad_module):
    OpcheckFastSoftMaxGradOperation = import_fastsoftmaxgrad_module['OpcheckFastSoftMaxGradOperation']
    op = OpcheckFastSoftMaxGradOperation()
    op.op_param = {}

    with patch.object(op, 'execute') as mock_execute:
        op.test_fastsoftmaxgrad()

    mock_execute.assert_called_once()
