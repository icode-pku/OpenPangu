import sys
from unittest.mock import patch, MagicMock

import pytest
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_activation_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.activation import OpcheckActivationOperation, ActivationType
    OpcheckActivationOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckActivationOperation": OpcheckActivationOperation,
        "ActivationType": ActivationType
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_activation_type_op_param_in_tensors_when_valid_input_then_correct_result(
                                                                                            import_activation_module):
    ActivationType = import_activation_module['ActivationType']

    test_cases = [
        {
            "activation_type": ActivationType.ACTIVATION_RELU.value,
            "op_param": {},
            "in_tensors": [torch.tensor([-1.0, 2.0, 0.0])],
            "expected_result": [torch.tensor([0.0, 2.0, 0.0])]
        }
    ]

    for case in test_cases:
        OpcheckActivationOperation = import_activation_module['OpcheckActivationOperation']
        op = OpcheckActivationOperation()
        op.op_param = {'activationType': case["activation_type"], **case["op_param"]}

        result = op.golden_calc(case["in_tensors"])

        assert torch.allclose(result[0], case["expected_result"][0], atol=1e-4), \
            f"Failed for activation_type={case['activation_type']}"


def test_test_given_op_param_when_valid_input_then_execute_successfully(import_activation_module):
    ActivationType = import_activation_module['ActivationType']
    test_cases = [
        ({'activationType': ActivationType.ACTIVATION_RELU.value}, True, True),
        ({'activationType': ActivationType.ACTIVATION_GELU.value}, True, True),
        ({'activationType': ActivationType.ACTIVATION_FAST_GELU.value}, True, True),
        ({'activationType': ActivationType.ACTIVATION_SWISH.value}, True, True),
        ({'activationType': ActivationType.ACTIVATION_LOG.value}, True, True),
        ({'activationType': ActivationType.ACTIVATION_SWIGLU_FORWARD.value}, True, True),
        ({'activationType': ActivationType.ACTIVATION_SWIGLU_BACKWARD.value}, True, True),
        ({}, False, False),
        ({'activationType': ActivationType.ACTIVATION_UNDEFINED.value}, False, False),
        ({'activationType': ActivationType.ACTIVATION_MAX.value}, False, False),
    ]
    
    for op_param, validate_param_return, expected_execute_call in test_cases:
        OpcheckActivationOperation = import_activation_module['OpcheckActivationOperation']
        op = OpcheckActivationOperation()
        op.op_param = op_param

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test()

        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()
