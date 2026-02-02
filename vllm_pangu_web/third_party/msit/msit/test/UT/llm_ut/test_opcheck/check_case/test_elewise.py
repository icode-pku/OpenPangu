import sys
from unittest.mock import patch, MagicMock

import pytest
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_elewise_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.elewise import OpcheckElewiseAddOperation, ElewiseType
    OpcheckElewiseAddOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckElewiseAddOperation": OpcheckElewiseAddOperation,
        "ElewiseType": ElewiseType
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_elewise_type_op_param_in_tensors_when_valid_input_then_correct_result(
                                                                                        import_elewise_module):
    ElewiseType = import_elewise_module['ElewiseType']

    test_cases = [
        {
            "elewise_type": ElewiseType.ELEWISE_MULS.value,
            "op_param": {'varAttr': 2.0},
            "in_tensors": [torch.tensor([1.0, 2.0])],
            "expected_result": [torch.tensor([2.0, 4.0])]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_COS.value,
            "op_param": {},
            "in_tensors": [torch.tensor([0.0, 1.0])],
            "expected_result": [torch.tensor([1.0, 0.5403])]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_SIN.value,
            "op_param": {},
            "in_tensors": [torch.tensor([0.0, 1.0])],
            "expected_result": [torch.tensor([0.0, 0.8415])]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_NEG.value,
            "op_param": {},
            "in_tensors": [torch.tensor([1.0, -2.0])],
            "expected_result": [torch.tensor([-1.0, 2.0])]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_QUANT.value,
            "op_param": {},
            "in_tensors": [torch.tensor([1.0, 2.0])],
            "expected_result": [torch.tensor([1, 2], dtype=torch.int8)]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_LOGICAL_NOT.value,
            "op_param": {},
            "in_tensors": [torch.tensor([True, False])],
            "expected_result": [torch.tensor([False, True])]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_ADD.value,
            "op_param": {},
            "in_tensors": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
            "expected_result": [torch.tensor([4.0, 6.0])]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_MUL.value,
            "op_param": {},
            "in_tensors": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
            "expected_result": [torch.tensor([3.0, 8.0])]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_REALDIV.value,
            "op_param": {},
            "in_tensors": [torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])],
            "expected_result": [torch.tensor([1.0, 1.0])]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_LOGICAL_AND.value,
            "op_param": {},
            "in_tensors": [torch.tensor([True, False]), torch.tensor([True, True])],
            "expected_result": [torch.tensor([1, 0], dtype=torch.int8)]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_LOGICAL_OR.value,
            "op_param": {},
            "in_tensors": [torch.tensor([True, False]), torch.tensor([False, False])],
            "expected_result": [torch.tensor([1, 0], dtype=torch.int8)]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_LESS.value,
            "op_param": {},
            "in_tensors": [torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0])],
            "expected_result": [torch.tensor([1, 0], dtype=torch.int8)]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_GREATER.value,
            "op_param": {},
            "in_tensors": [torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0])],
            "expected_result": [torch.tensor([0, 1], dtype=torch.int8)]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_SUB.value,
            "op_param": {},
            "in_tensors": [torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0])],
            "expected_result": [torch.tensor([-1.0, 1.0])]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_EQUAL.value,
            "op_param": {},
            "in_tensors": [torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0])],
            "expected_result": [torch.tensor([1, 0], dtype=torch.int8)]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_QUANT_PER_CHANNEL.value,
            "op_param": {},
            "in_tensors": [torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])],
            "expected_result": [torch.tensor([1, 2], dtype=torch.int8)]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_DEQUANT_PER_CHANNEL.value,
            "op_param": {},
            "in_tensors": [torch.tensor([1, 2], dtype=torch.int8), torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])],
            "expected_result": [torch.tensor([1.0, 2.0], dtype=torch.float16)]
        },
        {
            "elewise_type": ElewiseType.ELEWISE_TANH.value,
            "op_param": {},
            "in_tensors": [torch.tensor([0.0, 1.0])],
            "expected_result": [torch.tensor([0.0, 0.7616])]
        }
    ]

    for case in test_cases:
        OpcheckElewiseAddOperation = import_elewise_module['OpcheckElewiseAddOperation']
        op = OpcheckElewiseAddOperation()
        op.op_param = {'elewiseType': case["elewise_type"], **case["op_param"]}

        result = op.golden_calc(case["in_tensors"])

        for res, exp in zip(result, case["expected_result"]):
            assert torch.allclose(res, exp, atol=1e-4), \
                f"Failed for elewise_type={case['elewise_type']}"


def test_test_given_op_param_when_valid_input_then_execute_successfully(import_elewise_module):
    ElewiseType = import_elewise_module['ElewiseType']
    test_cases = [
        ({'elewiseType': ElewiseType.ELEWISE_CAST.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_MULS.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_COS.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_SIN.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_NEG.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_QUANT.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_LOGICAL_NOT.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_ADD.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_MUL.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_REALDIV.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_LOGICAL_AND.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_LOGICAL_OR.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_LESS.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_GREATER.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_SUB.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_EQUAL.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_QUANT_PER_CHANNEL.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_DEQUANT_PER_CHANNEL.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_DYNAMIC_QUANT.value}, True, True),
        ({'elewiseType': ElewiseType.ELEWISE_TANH.value}, True, True),
        ({}, False, False),
        ({'elewiseType': ElewiseType.ELEWISE_UNDEFINED.value}, False, False),
    ]
    
    for op_param, validate_param_return, expected_execute_call in test_cases:
        OpcheckElewiseAddOperation = import_elewise_module['OpcheckElewiseAddOperation']
        op = OpcheckElewiseAddOperation()
        op.op_param = op_param

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test()

        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()