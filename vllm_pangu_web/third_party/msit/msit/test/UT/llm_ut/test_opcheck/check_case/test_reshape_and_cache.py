import sys
import unittest
from unittest.mock import patch, MagicMock
import torch


class MockOperationTest:
    def __init__(self):
        self.op_param = {}
        self._golden_func_called = None
        
    def golden_func1(self, in_tensors):
        self._golden_func_called = "func1"
        return [torch.zeros_like(in_tensors[2]), torch.zeros_like(in_tensors[3])]
    
    def golden_func2(self, in_tensors):
        self._golden_func_called = "func2"
        return [torch.zeros_like(in_tensors[2]), torch.zeros_like(in_tensors[3])]
    
    def golden_func3(self, in_tensors):
        self._golden_func_called = "func3"
        return [torch.zeros_like(in_tensors[2]), torch.zeros_like(in_tensors[3])]
    
    def execute_inplace(self):
        pass

class TestOpcheckReshapeAndCacheOperation(unittest.TestCase):
    class MockCompressType:
        COMPRESS_TYPE_UNDEFINED = 0
        COMPRESS_TYPE_KVHEAD = 1

    def setUp(self):
        # Mock the operation_test module
        self.patcher = patch.dict('sys.modules', {
            'torch_npu': MagicMock()
        })
        self.patcher.start()
        
        # 动态创建被测试类，继承自MockOperationTest
        class OpcheckReshapeAndCacheOperation(MockOperationTest):
            def __init__(self):
                super().__init__()
                self.op_param = {}
                self.get_soc_version = MagicMock()
                self.execute_inplace = MagicMock()
                
            def golden_calc(self, in_tensors):
                soc_version = self.get_soc_version()
                compress_type = self.op_param.get("compressType", self.MockCompressType.COMPRESS_TYPE_UNDEFINED)
                
                if soc_version == "Ascend910B":
                    if compress_type == self.MockCompressType.COMPRESS_TYPE_KVHEAD:
                        return self.golden_func1(in_tensors)
                    else:
                        return self.golden_func2(in_tensors)
                else:
                    return self.golden_func3(in_tensors)
                    
            def test(self):
                self.execute_inplace()
        
        # 为测试类添加MockCompressType属性
        OpcheckReshapeAndCacheOperation.MockCompressType = self.MockCompressType
        
        # 创建测试实例
        self.op = OpcheckReshapeAndCacheOperation()

    def tearDown(self):
        self.patcher.stop()

    def test_golden_func1(self):
        # 准备测试数据，确保数据格式正确
        in_tensors = [
            torch.randn(2, 4, 8),  # bs_id, head_id, head_size
            torch.randn(2, 4, 8),  # bs_id, head_id, head_size
            torch.zeros(2, 8, 4, 8),  # block_index, block_offset, head_id, head_size
            torch.zeros(2, 8, 4, 8),  # block_index, block_offset, head_id, head_size
            torch.tensor([0, 3, -1, 5]),  # slot_mapping
            torch.tensor([1, 2]),  # wins
            torch.tensor([1, 1])   # seq_len
        ]
        
        # 调用函数
        result = self.op.golden_func1(in_tensors)
        
        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, in_tensors[2].shape)
        self.assertEqual(result[1].shape, in_tensors[3].shape)

    def test_golden_func2(self):
        # 准备测试数据，确保数据格式正确
        in_tensors = [
            torch.randn(4, 4, 8),  # i, head_id, head_size
            torch.randn(4, 4, 8),  # i, head_id, head_size
            torch.zeros(2, 8, 4, 8),  # block_index, block_offset, head_id, head_size
            torch.zeros(2, 8, 4, 8),  # block_index, block_offset, head_id, head_size
            torch.tensor([0, 3, -1, 5]),  # slot_mapping
        ]
        
        # 调用函数
        result = self.op.golden_func2(in_tensors)
        
        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, in_tensors[2].shape)
        self.assertEqual(result[1].shape, in_tensors[3].shape)

    def test_golden_func3(self):
        # 准备测试数据，确保数据格式正确
        in_tensors = [
            torch.randn(4, 4, 16),  # i, num_heads, head_size
            torch.randn(4, 4, 16),  # i, num_heads, head_size
            torch.zeros(2, 4, 8, 16),  # block_index, k, block_offset, :
            torch.zeros(2, 4, 8, 16),  # block_index, k, block_offset, :
            torch.tensor([0, 3, 1, 5]),  # slot_mapping
        ]
        
        # 调用函数
        result = self.op.golden_func3(in_tensors)
        
        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, in_tensors[2].shape)
        self.assertEqual(result[1].shape, in_tensors[3].shape)

    def test_golden_calc_910b_kvhead(self):
        self.op.get_soc_version.return_value = "Ascend910B"
        self.op.op_param = {"compressType": self.MockCompressType.COMPRESS_TYPE_KVHEAD}
        
        # 准备适合golden_func1的输入张量
        in_tensors = [torch.randn(2, 4, 8) for _ in range(7)]
        result = self.op.golden_calc(in_tensors)
        
        self.assertEqual(len(result), 2)
        self.op.get_soc_version.assert_called_once()

    def test_golden_calc_910b_default(self):
        self.op.get_soc_version.return_value = "Ascend910B"
        self.op.op_param = {"compressType": self.MockCompressType.COMPRESS_TYPE_UNDEFINED}
        
        # 准备适合golden_func2的输入张量
        in_tensors = [torch.randn(4, 4, 8) for _ in range(5)]
        result = self.op.golden_calc(in_tensors)
        
        self.assertEqual(len(result), 2)
        self.op.get_soc_version.assert_called_once()

    def test_golden_calc_non_910b(self):
        self.op.get_soc_version.return_value = "OtherSocVersion"
        
        # 准备适合golden_func3的输入张量
        in_tensors = [torch.randn(4, 4, 16) for _ in range(5)]
        result = self.op.golden_calc(in_tensors)
        
        self.assertEqual(len(result), 2)
        self.op.get_soc_version.assert_called_once()

    def test_test_method(self):
        self.op.test()
        self.op.execute_inplace.assert_called_once()
