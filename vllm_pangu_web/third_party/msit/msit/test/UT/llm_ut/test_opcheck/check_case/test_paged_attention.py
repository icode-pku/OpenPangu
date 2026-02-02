import sys
from unittest.mock import patch, MagicMock
import pytest
import torch


from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_paged_attention_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.paged_attention import OpcheckPagedAttentionAttentionOperation, MaskType, QuantType
    OpcheckPagedAttentionAttentionOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckPagedAttentionAttentionOperation": OpcheckPagedAttentionAttentionOperation,
        "MaskType": MaskType,
        "QuantType": QuantType
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


# Test cases for mask_nz_2_nd
def test_mask_nz_2_nd_given_mask_and_context_lens_when_mask_type_alibi_then_correct_shape(
                                                                                    import_paged_attention_module):
    MaskType = import_paged_attention_module['MaskType']
    OpcheckPagedAttentionAttentionOperation = import_paged_attention_module['OpcheckPagedAttentionAttentionOperation']
    mask = torch.randn(4, 8, 16, 32)
    mask_type = MaskType.MASK_TYPE_ALIBI.value
    context_lens = [4, 8, 12, 16]
    head_num = 4
    result_mask = OpcheckPagedAttentionAttentionOperation.mask_nz_2_nd(mask, mask_type, context_lens, head_num)
    assert result_mask.shape == (1, head_num, 16, 16 * 16)


def test_mask_nz_2_nd_given_mask_and_context_lens_when_mask_type_norm_then_correct_shape(import_paged_attention_module):
    MaskType = import_paged_attention_module['MaskType']
    OpcheckPagedAttentionAttentionOperation = import_paged_attention_module['OpcheckPagedAttentionAttentionOperation']
    mask = torch.randn(4, 8, 16, 32)
    mask_type = MaskType.MASK_TYPE_NORM.value
    context_lens = [4, 8, 12, 16]
    head_num = 4
    result_mask = OpcheckPagedAttentionAttentionOperation.mask_nz_2_nd(mask, mask_type, context_lens, head_num)
    assert result_mask.shape == (4, 16, 16 * 16)


def test_mask_nz_2_nd_given_mask_and_context_lens_when_mask_type_undefined_then_correct_shape(
                                                                                        import_paged_attention_module):
    MaskType = import_paged_attention_module['MaskType']
    OpcheckPagedAttentionAttentionOperation = import_paged_attention_module['OpcheckPagedAttentionAttentionOperation']
    mask = torch.randn(4, 8, 16, 32)
    mask_type = MaskType.UNDEFINED.value
    context_lens = [4, 8, 12, 16]
    head_num = 4
    result_mask = OpcheckPagedAttentionAttentionOperation.mask_nz_2_nd(mask, mask_type, context_lens, head_num)
    assert result_mask.shape == (4, 16, 16 * 16)


# # Test cases for get_quant_param
def test_get_quant_param_given_quant_type_has_quant_offset_when_quant_type_undefined_then_correct_values(
                                                                                    import_paged_attention_module):
    QuantType = import_paged_attention_module['QuantType']
    OpcheckPagedAttentionAttentionOperation = import_paged_attention_module['OpcheckPagedAttentionAttentionOperation']
    op = OpcheckPagedAttentionAttentionOperation()
    quant_type = QuantType.TYPE_QUANT_UNDEFINED.value
    has_quant_offset = False
    is_int8_flag, has_bias, fp32_input = op.get_quant_param(quant_type, has_quant_offset)
    assert is_int8_flag is False
    assert has_bias is False
    assert fp32_input == [None, None, None, None]


def test_ref_masked_attention_given_masked_attention_input_alibi_bias_when_quant_type_dequant_fusion_then_correct_shape(
        import_paged_attention_module):
    OpcheckPagedAttentionAttentionOperation = import_paged_attention_module['OpcheckPagedAttentionAttentionOperation']
    op = OpcheckPagedAttentionAttentionOperation()
    masked_attention_input = [torch.randn(1, 8, 16), torch.randn(16, 4, 16), torch.randn(16, 4, 16), 1.0]
    alibi_bias = torch.randn(8, 1, 16)
    out = op.ref_masked_attention(masked_attention_input, alibi_bias)
    assert out.shape == (1, 8, 16)
