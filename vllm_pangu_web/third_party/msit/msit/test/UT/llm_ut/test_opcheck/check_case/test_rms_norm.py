import sys
from unittest.mock import patch, MagicMock

import pytest


from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_rms_norm_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.rms_norm import OpcheckRmsNormOperation, RmsNormType, QuantType
    OpcheckRmsNormOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckRmsNormOperation": OpcheckRmsNormOperation,
        "RmsNormType": RmsNormType,
        "QuantType": QuantType,
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_validate_rmsnorm_param_given_valid_layer_type_when_validate_then_correct_result(import_rms_norm_module):
    OpcheckRmsNormOperation = import_rms_norm_module['OpcheckRmsNormOperation']
    RmsNormType = import_rms_norm_module['RmsNormType']
    QuantType = import_rms_norm_module['QuantType']
    op = OpcheckRmsNormOperation()
    op.op_param = {'layerType': RmsNormType.RMS_NORM_NORM.value, 'normParam': {'quantType': QuantType.QUANT_INT8.value}}

    result = op.validate_rmsnorm_param(RmsNormType.RMS_NORM_NORM.value)

    assert result['quantType'] == QuantType.QUANT_INT8.value


def test_test_when_valid_input(import_rms_norm_module):
    OpcheckRmsNormOperation = import_rms_norm_module['OpcheckRmsNormOperation']
    RmsNormType = import_rms_norm_module['RmsNormType']
    QuantType = import_rms_norm_module['QuantType']
    test_cases = [
        ({'layerType': RmsNormType.RMS_NORM_NORM.value,
        'normParam': {'quantType': QuantType.QUANT_INT8.value, 'epsilon': 1e-5}}, True),
        ({'layerType': RmsNormType.RMS_NORM_PRE_NORM.value,
          'preNormParam': {'quantType': QuantType.QUANT_INT8.value, 'epsilon': 1e-5}}, True),
        ({'layerType': RmsNormType.RMS_NORM_POST_NORM.value,
          'postNormParam': {'quantType': QuantType.QUANT_UNDEFINED.value, 'epsilon': 1e-5}}, True),
    ]
    for op_param, expected_execute_call in test_cases:
        op = OpcheckRmsNormOperation()
        op.op_param = op_param
        with patch.object(op, 'validate_param', return_value=True):
            with patch.object(op, 'execute') as mock_execute:
                op.test()
                if expected_execute_call:
                    mock_execute.assert_called_once()
                else:
                    mock_execute.assert_not_called()
