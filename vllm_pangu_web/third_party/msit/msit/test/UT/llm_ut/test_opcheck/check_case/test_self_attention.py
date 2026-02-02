import sys
from dataclasses import dataclass
from unittest.mock import MagicMock
import pytest
import torch


# Mocking the OperationTest class to avoid errors
from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_self_attention_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.self_attention import OpcheckUnpadSelfAttentionOperation, MaskType, ClampType
    OpcheckUnpadSelfAttentionOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckUnpadSelfAttentionOperation": OpcheckUnpadSelfAttentionOperation,
        "MaskType": MaskType,
        "ClampType": ClampType
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


@dataclass
class AttentionMaskParams:
    attention_mask: torch.Tensor
    seq_len: list
    expected_shape: tuple


@pytest.mark.parametrize("params", [
    AttentionMaskParams(torch.randn(4, 8, 16, 32), [4, 8, 12, 16], (64, 8, 16)),
])
def test_attention_mask_nz_2_nd_when_valid_input_then_correct_shape(params, import_self_attention_module):
    OpcheckUnpadSelfAttentionOperation = import_self_attention_module["OpcheckUnpadSelfAttentionOperation"]
    result_mask = OpcheckUnpadSelfAttentionOperation.attention_mask_nz_2_nd(params.attention_mask, params.seq_len)
    assert result_mask.shape == params.expected_shape


@dataclass
class GroupMatmulParams:
    head_num: int
    group_num: int
    in_a: torch.Tensor
    in_b: torch.Tensor
    expected_shape: tuple


@pytest.mark.parametrize("params", [
    GroupMatmulParams(8, 4, torch.randn(8, 16, 32), torch.randn(4, 32, 16), (8, 16, 16)),
])
def test_group_matmul_when_valid_input_then_correct_shape(params, import_self_attention_module):
    OpcheckUnpadSelfAttentionOperation = import_self_attention_module["OpcheckUnpadSelfAttentionOperation"]
    score = OpcheckUnpadSelfAttentionOperation.group_matmul(params.head_num, params.group_num, params.in_a, params.in_b)
    assert score.shape == params.expected_shape


@pytest.mark.parametrize("params", [
    GroupMatmulParams(8, 0, torch.randn(8, 16, 32), torch.randn(4, 32, 16), None),
])
def test_group_matmul_when_zero_division_then_raise_error(params, import_self_attention_module):
    OpcheckUnpadSelfAttentionOperation = import_self_attention_module["OpcheckUnpadSelfAttentionOperation"]
    with pytest.raises(RuntimeError):
        OpcheckUnpadSelfAttentionOperation.group_matmul(params.head_num, params.group_num, params.in_a, params.in_b)


@dataclass
class ReshapeQkvParams:
    qkv: torch.Tensor
    is_pa: bool
    expected_shape: tuple


@pytest.mark.parametrize("params", [
    ReshapeQkvParams(torch.randn(4, 8, 16, 32), True, (32, 512)),
    ReshapeQkvParams(torch.randn(4, 8, 16), True, (4, 128)),
])
def test_reshape_qkv_when_valid_input_then_correct_shape(params,import_self_attention_module):
    OpcheckUnpadSelfAttentionOperation = import_self_attention_module["OpcheckUnpadSelfAttentionOperation"]
    result_qkv = OpcheckUnpadSelfAttentionOperation.reshape_qkv(params.qkv, params.is_pa)
    assert result_qkv.shape == params.expected_shape


@dataclass
class GetQkvParams:
    in_tensors: list
    expected_shapes: list


@pytest.mark.parametrize("params", [
    GetQkvParams([torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32)],
                 [(32, 512), (32, 512), (32, 512)]),
])
def test_get_qkv_when_valid_input_then_correct_shape(params, import_self_attention_module):
    OpcheckUnpadSelfAttentionOperation = import_self_attention_module["OpcheckUnpadSelfAttentionOperation"]
    q, k, v = OpcheckUnpadSelfAttentionOperation.get_qkv(params.in_tensors)
    assert q.shape == params.expected_shapes[0]
    assert k.shape == params.expected_shapes[1]
    assert v.shape == params.expected_shapes[2]


@dataclass
class GetOutSubParams:
    head_info: list
    q_s: int
    score: torch.Tensor
    v_slice: torch.Tensor
    expected_shape: tuple


@pytest.mark.parametrize("params", [
    GetOutSubParams([8, 16, 4], 4, torch.randn(8, 4, 16), torch.randn(4, 16, 16), (4, 8, 16)),
])
def test_get_out_sub_when_valid_input_then_correct_shape(params, import_self_attention_module):
    _p = None
    OpcheckUnpadSelfAttentionOperation = import_self_attention_module["OpcheckUnpadSelfAttentionOperation"]
    out_sub, _p = OpcheckUnpadSelfAttentionOperation.get_out_sub(params.head_info, params.q_s, params.score,
                                                                 params.v_slice, _p)
    assert out_sub.shape == params.expected_shape


@dataclass
class GetBatchStatusParams:
    in_tensors: list
    seq_len: list
    expected_batch_status: list


@pytest.mark.parametrize("params", [
    GetBatchStatusParams(
        [torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32),
         [4, 8, 12, 16], range(4)], [4, 8, 12, 16], [0, 1, 2, 3]),
    GetBatchStatusParams(
        [torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32),
         [4, 8, 12, 16]], [4, 8, 12, 16], [0, 1, 2, 3]),
])
def test_get_batch_status_when_valid_input_then_correct_shape(params, import_self_attention_module):
    OpcheckUnpadSelfAttentionOperation = import_self_attention_module["OpcheckUnpadSelfAttentionOperation"]
    op = OpcheckUnpadSelfAttentionOperation()
    batch_status = op.get_batch_status(params.in_tensors, params.seq_len)
    assert list(batch_status) == params.expected_batch_status


@dataclass
class GetPostMaskCoffParams:
    data_type: torch.dtype
    mask_type: int
    expected_value: float


def test_get_post_mask_coff_when_valid_input_then_correct_value(import_self_attention_module):
    MaskType = import_self_attention_module["MaskType"]
    test_cases = [
        GetPostMaskCoffParams(torch.float16, MaskType.MASK_TYPE_UNDEFINED.value, 1.0),
        GetPostMaskCoffParams(torch.bfloat16, MaskType.MASK_TYPE_ALIBI.value, 1.0),
        GetPostMaskCoffParams(torch.float32, MaskType.MASK_TYPE_ALIBI.value, 1.0),
        GetPostMaskCoffParams(torch.float32, MaskType.MASK_TYPE_UNDEFINED.value, -3e38),
    ]
    for params in test_cases:
        OpcheckUnpadSelfAttentionOperation = import_self_attention_module["OpcheckUnpadSelfAttentionOperation"]
        op = OpcheckUnpadSelfAttentionOperation()
        op.op_param['maskType'] = params.mask_type
        post_mask_coff = op.get_post_mask_coff(params.data_type)
        assert post_mask_coff == params.expected_value


@dataclass
class GetAttentionParamsParams:
    q: torch.Tensor
    q_scale: float
    qk_scale: float
    head_num: int
    kv_head_num: int
    expected_params: tuple


@pytest.mark.parametrize("params", [
    GetAttentionParamsParams(torch.randn(4, 8, 16), 2.0, 3.0, 4, 2, [2.0, 3.0, [4, 2, 2], torch.float32, 4]),
])
def test_get_attention_params_when_valid_input_then_correct_values(params, import_self_attention_module):
    OpcheckUnpadSelfAttentionOperation = import_self_attention_module["OpcheckUnpadSelfAttentionOperation"]
    op = OpcheckUnpadSelfAttentionOperation()
    op.op_param['qScale'] = params.q_scale
    op.op_param['qkScale'] = params.qk_scale
    op.op_param['headNum'] = params.head_num
    op.op_param['kvHeadNum'] = params.kv_head_num
    result_params = op.get_attention_params(params.q)
    assert result_params == list(params.expected_params)


@dataclass
class GetClampedScoreParams:
    score: torch.Tensor
    clamp_type: int
    clamp_min: float
    clamp_max: float
    expected_min: float
    expected_max: float



def test_get_clamped_score_when_valid_input_then_correct_values(import_self_attention_module):
    ClampType = import_self_attention_module["ClampType"]
    testcases = [
        GetClampedScoreParams(torch.randn(4, 8, 16), ClampType.CLAMP_TYPE_MIN_MAX.value, -1.0, 1.0, -1.0, 1.0),
        GetClampedScoreParams(torch.randn(4, 8, 16), ClampType.CLAMP_TYPE_UNDEFINED.value, 0.0, 0.0, float('-inf'),
                          float('inf')),
    ]
    for params in testcases:
        OpcheckUnpadSelfAttentionOperation = import_self_attention_module["OpcheckUnpadSelfAttentionOperation"]
        op = OpcheckUnpadSelfAttentionOperation()
        op.op_param['clampType'] = params.clamp_type
        op.op_param['clampMin'] = params.clamp_min
        op.op_param['clampMax'] = params.clamp_max
        clamped_score = op.get_clamped_score(params.score)
        assert torch.all(clamped_score >= params.expected_min)
        assert torch.all(clamped_score <= params.expected_max)
