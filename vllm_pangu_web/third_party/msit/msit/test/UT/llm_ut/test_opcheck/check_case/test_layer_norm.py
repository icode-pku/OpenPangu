import sys
from unittest.mock import patch, MagicMock

import pytest
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_layer_norm_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.layer_norm import OpcheckLayerNormOperation, LayerNormType
    from msit_llm.common.log import logger
    OpcheckLayerNormOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckLayerNormOperation": OpcheckLayerNormOperation,
        "LayerNormType": LayerNormType,
        "logger": logger
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


# Mock the logger to capture log messages
@pytest.fixture
def mock_logger(import_layer_norm_module):
    logger = import_layer_norm_module['logger']
    with patch.object(logger, 'info') as mock_info, patch.object(logger, 'debug') as mock_debug:
        yield mock_info, mock_debug


def test_test_when_valid_input(import_layer_norm_module):
    LayerNormType = import_layer_norm_module['LayerNormType']
    test_cases = [
         (LayerNormType.LAYER_NORM_NROM.value, True, True),
        (LayerNormType.LAYER_NORM_PRENORM.value, True, True),
        (LayerNormType.LAYER_NORM_POSTNORM.value, True, True),
        (LayerNormType.LAYER_NORM_UNDEFINED.value, False, False),
    ]
    for layer_type, validate_param_return, expected_execute_call in test_cases:
        OpcheckLayerNormOperation = import_layer_norm_module['OpcheckLayerNormOperation']
        op = OpcheckLayerNormOperation()
        op.op_param = {'layerType': layer_type}

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test()

        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()
