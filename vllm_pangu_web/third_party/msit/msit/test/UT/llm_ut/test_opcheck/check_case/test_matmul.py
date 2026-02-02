import sys
from unittest.mock import patch, MagicMock
import pytest
import torch
import numpy as np


from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_matmul_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.matmul import OpcheckMatmulOperation
    OpcheckMatmulOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckMatmulOperation": OpcheckMatmulOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_flp_given_transpose_a_false_and_transpose_b_true_when_2d_tensors_then_correct_result(
                                                                                                import_matmul_module):
    OpcheckMatmulOperation = import_matmul_module["OpcheckMatmulOperation"]
    op = OpcheckMatmulOperation()
    transpose_a = False
    transpose_b = True
    in_tensor_0 = torch.tensor([[1, 2], [3, 4]])
    in_tensor_1 = torch.tensor([[5, 6], [7, 8]])

    result = op.golden_flp(transpose_a, transpose_b, in_tensor_0, in_tensor_1)

    expected = torch.tensor([[17, 23], [39, 53]])
    assert torch.allclose(result, expected)


def test_golden_flp_given_transpose_a_false_and_transpose_b_false_when_2d_tensors_then_correct_result(
                                                                                                import_matmul_module):
    OpcheckMatmulOperation = import_matmul_module["OpcheckMatmulOperation"]
    op = OpcheckMatmulOperation()
    transpose_a = False
    transpose_b = False
    in_tensor_0 = torch.tensor([[1, 2], [3, 4]])
    in_tensor_1 = torch.tensor([[5, 6], [7, 8]])

    result = op.golden_flp(transpose_a, transpose_b, in_tensor_0, in_tensor_1)

    expected = torch.tensor([[19, 22], [43, 50]])
    assert torch.allclose(result, expected)


def test_golden_calc_given_transpose_a_false_and_transpose_b_true_when_2d_tensors_then_correct_result(
                                                                                                import_matmul_module):
    OpcheckMatmulOperation = import_matmul_module["OpcheckMatmulOperation"]
    op = OpcheckMatmulOperation()
    op.op_param = {"transposeA": False, "transposeB": True}
    in_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])]

    result = op.golden_calc(in_tensors)

    expected = torch.tensor([[17, 23], [39, 53]]).half()
    assert torch.allclose(result[0], expected)


def test_golden_calc_given_transpose_a_false_and_transpose_b_false_when_2d_tensors_then_correct_result(
                                                                                                import_matmul_module):
    OpcheckMatmulOperation = import_matmul_module["OpcheckMatmulOperation"]
    op = OpcheckMatmulOperation()
    op.op_param = {"transposeA": False, "transposeB": False}
    in_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])]

    result = op.golden_calc(in_tensors)

    expected = torch.tensor([[19, 22], [43, 50]]).half()
    assert torch.allclose(result[0], expected)
