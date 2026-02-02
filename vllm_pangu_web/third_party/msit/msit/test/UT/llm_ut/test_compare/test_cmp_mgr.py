import unittest
from unittest.mock import patch, MagicMock

import pytest
import torch

from msit_llm.common.log import logger
from msit_llm.compare.cmp_mgr import CompareMgr


@pytest.fixture()
def mock_logger():
    with patch.object(logger, 'error') as mock_error, patch.object(logger, 'debug') as mock_debug:
        yield mock_error, mock_debug


def test_filter_rope_my_tensor_paths_valid(mock_logger):
    mock_error, mock_debug = mock_logger
    my_tensor_paths = [
        'path/intensor4.bin',
        'path/intensor5.bin',
        'path/intensor3.bin',
        'path/intensor2.bin',
        'path/intensor1.bin'
    ]
    seqlen_path, valid_paths = CompareMgr._filter_rope_my_tensor_paths(my_tensor_paths)
    assert seqlen_path == 'path/intensor4.bin'
    assert valid_paths == [
        'path/intensor2.bin',
        'path/intensor3.bin'
    ]
    mock_error.assert_not_called()
    mock_debug.assert_not_called()


def test_filter_rope_my_tensor_paths_when_invalid_file_numbers(mock_logger):
    mock_error, mock_debug = mock_logger
    my_tensor_paths = [
        'path/intensor4.bin',
        'path/intensor5.bin'
    ]
    seqlen_path, valid_paths = CompareMgr._filter_rope_my_tensor_paths(my_tensor_paths)
    assert valid_paths == my_tensor_paths
    mock_debug.assert_called_once_with(f"Expected 5 tensors for RopeOperation but found {len(my_tensor_paths)}.")
    mock_error.assert_not_called()


def test_filter_rope_my_tensor_paths_when_invalid_file(mock_logger):
    mock_error, mock_debug = mock_logger
    my_tensor_paths = [
        'path/intensor4.bin',
        'path/intensor5.bin',
        'path/intensor6.bin',
        'path/intensor2.bin',
        'path/intensor1.bin'
    ]
    seqlen_path, valid_paths = CompareMgr._filter_rope_my_tensor_paths(my_tensor_paths)
    assert valid_paths == my_tensor_paths
    mock_debug.assert_called_once()
    mock_error.assert_not_called()


def test_get_rope_type_when_given_intensor2(mock_logger):
    mock_error, mock_debug = mock_logger
    tensor_path = 'path/intensor2.bin'
    rope_type = CompareMgr._get_rope_type(tensor_path)
    assert rope_type == 0
    mock_error.assert_not_called()
    mock_debug.assert_not_called()


def test_get_rope_type_when_given_intensor3(mock_logger):
    mock_error, mock_debug = mock_logger
    tensor_path = 'path/intensor3.bin'
    rope_type = CompareMgr._get_rope_type(tensor_path)
    assert rope_type == 1
    mock_error.assert_not_called()
    mock_debug.assert_not_called()


def test_get_rope_type_when_invalid(mock_logger):
    mock_error, mock_debug = mock_logger
    tensor_path = 'path/intensor1.bin'
    rope_type = CompareMgr._get_rope_type(tensor_path)
    assert rope_type == -1
    mock_debug.assert_called_once_with(f"Failed to get rope_type from {tensor_path}.")
    mock_error.assert_not_called()


def test_slice_tensor_by_seq_len_4d_valid(mock_logger):
    mock_error, mock_debug = mock_logger
    tensor = torch.randn(1, 3, 5, 4)
    golden_tensor_datas = [tensor]
    seq_len = 3
    rope_type = 0

    sliced_tensor = CompareMgr._slice_tensor_by_seq_len(golden_tensor_datas, seq_len, rope_type)
    expected_sliced_tensor = tensor[:, :, seq_len - 1, :].squeeze(0)
    assert torch.equal(sliced_tensor[0], expected_sliced_tensor)
    mock_error.assert_not_called()
    mock_debug.assert_not_called()


def test_slice_tensor_by_seq_len_3d_valid(mock_logger):
    mock_error, mock_debug = mock_logger
    tensor = torch.randn(4, 3, 2)
    golden_tensor_datas = [tensor]
    seq_len = 2
    rope_type = 1

    sliced_tensor = CompareMgr._slice_tensor_by_seq_len(golden_tensor_datas, seq_len, rope_type)
    expected_sliced_tensor = tensor[seq_len - 1, :, rope_type].unsqueeze(0)
    assert torch.equal(sliced_tensor[0], expected_sliced_tensor)
    mock_error.assert_not_called()
    mock_debug.assert_not_called()


def test_slice_tensor_by_seq_len_4d_with_invalid_seq_len(mock_logger):
    mock_error, mock_debug = mock_logger
    tensor = torch.randn(1, 3, 5, 4)
    golden_tensor_datas = [tensor]
    seq_len = 6
    rope_type = 0

    sliced_tensor = CompareMgr._slice_tensor_by_seq_len(golden_tensor_datas, seq_len, rope_type)
    assert sliced_tensor == golden_tensor_datas
    mock_error.assert_called_once_with(f"seqLen is out of bounds for tensor with shape {tensor.shape}")
    mock_debug.assert_not_called()


def test_slice_tensor_by_seq_len_3d_with_invalid_seq_len(mock_logger):
    mock_error, mock_debug = mock_logger
    tensor = torch.randn(1, 3, 5, 4)
    golden_tensor_datas = [tensor]
    seq_len = None
    rope_type = 0

    sliced_tensor = CompareMgr._slice_tensor_by_seq_len(golden_tensor_datas, seq_len, rope_type)
    assert sliced_tensor == golden_tensor_datas
    mock_debug.assert_called_once_with("seq_len is None, skipping slicing.")
    mock_error.assert_not_called()


def test_slice_tensor_by_seq_len_with_invalid_dim(mock_logger):
    mock_error, mock_debug = mock_logger
    tensor = torch.randn(1, 3, 5, 6, 7)
    golden_tensor_datas = [tensor]
    seq_len = 2
    rope_type = 0

    sliced_tensor = CompareMgr._slice_tensor_by_seq_len(golden_tensor_datas, seq_len, rope_type)
    assert sliced_tensor == golden_tensor_datas
    mock_debug.assert_called_once_with(f"Unsupported tensor with dimensions {tensor.ndimension()}. Expected 3 or 4 "
                                       f"dimensions.")
    mock_error.assert_not_called()


def test_remove_adjacent_repeated_columns_valid(mock_logger):
    mock_error, mock_debug = mock_logger
    tensor = torch.tensor([[1, 1, 2, 2]], dtype=torch.float16)
    my_tensor_datas = [tensor]

    res_tensor_datas = CompareMgr._remove_adjacent_repeated_columns(my_tensor_datas)
    expected_tensor_datas = torch.tensor([[1, 2]], dtype=torch.float16)
    assert torch.equal(res_tensor_datas[0], expected_tensor_datas)
    mock_error.assert_not_called()
    mock_debug.assert_not_called()


def test_remove_adjacent_repeated_columns_with_invalid_tensor(mock_logger):
    mock_error, mock_debug = mock_logger
    tensor = torch.tensor([[1, 1, 2, 3]], dtype=torch.float16)
    my_tensor_datas = [tensor]

    res_tensor_datas = CompareMgr._remove_adjacent_repeated_columns(my_tensor_datas)
    assert res_tensor_datas == my_tensor_datas
    mock_error.assert_not_called()
    mock_debug.assert_called_once_with(f"There are no adjacent repeated columns in the tensor {my_tensor_datas}.")


def test_remove_adjacent_repeated_columns_with_invalid_shape(mock_logger):
    mock_error, mock_debug = mock_logger
    tensor = torch.tensor([[1, 1, 2, 3, 5]], dtype=torch.float16)
    my_tensor_datas = [tensor]

    res_tensor_datas = CompareMgr._remove_adjacent_repeated_columns(my_tensor_datas)
    assert res_tensor_datas == my_tensor_datas
    mock_error.assert_not_called()
    mock_debug.assert_called_once_with(f"Unexpected tensor shape {tensor.shape} to check whether has adjacent "
                                       f"repeated columns.")


class TestCompareMgr(unittest.TestCase):
    def setUp(self):
        # 绕过 __init__ 创建实例，不触发复杂逻辑
        self.mgr = CompareMgr.__new__(CompareMgr)
        self.mgr.compared_result = []
        self.mgr.args = MagicMock()
        self.mgr.perform_comparison = MagicMock()

    def test_recount_data_id_for_multiprocess_basic(self):
        """测试 data_id 从指定起点递增"""
        self.mgr.compared_result = [
            {"name": "a", "data_id": 2},
            {"name": "b", "data_id": 5},
            {"name": "c", "data_id": 1}
        ]
        self.mgr.recount_data_id_for_multiprocess(5)

        expected = [
            {"name": "a", "data_id": 5},
            {"name": "b", "data_id": 6},
            {"name": "c", "data_id": 7}
        ]

        assert self.mgr.compared_result == expected

    def test_recount_data_id_for_multiprocess_empty(self):
        """测试空列表不会报错"""
        self.mgr.compared_result = []
        self.mgr.recount_data_id_for_multiprocess(0)
        self.assertEqual(self.mgr.compared_result, [])

    def test_single_op_map_compare_with_stats_true(self):
        """当 self.stats=True 时应以 is_statistics=True 调用 perform_comparison"""
        self.mgr.recount_data_id_for_multiprocess = MagicMock()
        self.mgr.stats = True
        self.mgr.single_op_map_compare("g", "m", {"op": 1}, 10)
        self.assertEqual(self.mgr.compared_result, [])
        self.mgr.perform_comparison.assert_called_once_with("g", "m", {"op": 1}, is_statistics=True)
        self.mgr.recount_data_id_for_multiprocess.assert_called_once_with(10)

    def test_single_op_map_compare_with_stats_false(self):
        """当 self.stats=False 时应以 is_statistics=False 调用 perform_comparison"""
        self.mgr.recount_data_id_for_multiprocess = MagicMock()
        self.mgr.stats = False
        self.mgr.single_op_map_compare("g", "m", {"op": 2}, 5)
        self.mgr.perform_comparison.assert_called_once_with("g", "m", {"op": 2}, is_statistics=False)
        self.mgr.recount_data_id_for_multiprocess.assert_called_once_with(5)
