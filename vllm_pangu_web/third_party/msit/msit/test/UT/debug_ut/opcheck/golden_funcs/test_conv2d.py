import sys
from unittest.mock import patch, MagicMock
import unittest
import numpy as np


class TestConv2dOperationOtherFuncs(unittest.TestCase):
    def setUp(self):
        self.backup_modules = {}
        self.modules_to_mock = [
            'tensorflow',
            'tensorflow.compat',
            'tensorflow.compat.v1',
        ]
        for mod in self.modules_to_mock:
            if mod in sys.modules:
                self.backup_modules[mod] = sys.modules[mod]
            sys.modules[mod] = MagicMock()
        from msit_opcheck.golden_funcs.conv2d import Conv2dOperation
        from msit_opcheck.operation_test import OperationTest
        self.OperationTest = OperationTest
        self.Conv2dOperation = Conv2dOperation

    def tearDown(self):
        for mod in self.modules_to_mock:
            if mod in self.backup_modules:
                sys.modules[mod] = self.backup_modules[mod]
            else:
                del sys.modules[mod] 
        for mod in list(sys.modules.keys()):
            if mod.startswith("msit_opcheck.golden_funcs"):
                del sys.modules[mod]
    
    def test_basic_conv_params(self):
        """Test with basic convolution parameters (NHWC format)"""
        with patch.object(self.OperationTest, "__init__", return_value=None):
            conv2d_op = self.Conv2dOperation()
            conv2d_op.op_param = {
                'attr': [
                    {'key': 'strides', 'value': {'list': {'i': [1, 2, 2, 1]}}},
                    {'key': 'pads', 'value': {'list': {'i': [0, 1, 0, 1]}}},
                    {'key': 'dilations', 'value': {'list': {'i': [1, 1, 1, 1]}}},
                    {'key': 'groups', 'value': {'i': 1}},
                    {'key': 'data_format', 'value': {'s': 'NHWC'}}
                ]
            }
            params, groups = conv2d_op.get_conv_params()
            self.assertEqual(params['dilations'], [1, 1, 1, 1])
            self.assertEqual(params['pads'], [0, 1, 0, 1])
            self.assertEqual(params['strideh'], 2)
            self.assertEqual(params['stridew'], 2)
            self.assertEqual(groups, 1)
    
    def test_conv2d_with_groups(self):
        batch_size = 2
        height = 4
        width = 4
        kernel_size = 3
        channels = 4
        out_channels = 2        
        x = np.random.randn(batch_size, height, width, channels).astype(np.float32) # input tensor
        base_conv_params = {
                    'pads': (1, 1, 1, 1),  # (top, bottom, left, right)
                    'strideh': 1,
                    'stridew': 1,
                    'dilations': (1, 1)
        }
        with patch.object(self.OperationTest, "__init__", return_value=None):
            # test group is 1
            conv_filter = np.random.randn(kernel_size, kernel_size, channels, out_channels).astype(np.float32)
            bias = np.random.randn(out_channels).astype(np.float32)
            basic_result = self.Conv2dOperation.conv2d_with_groups(x, conv_filter, 1, bias, base_conv_params)
            assert 'mock.compat.v1.Session().__enter__().run()' in str(basic_result)
    
    def test_conv2d(self):
        x = np.random.rand(1, 5, 5, 3).astype(np.float32)  # batch=1, height=5, width=5, channels=3
        conv_filter = np.random.rand(3, 3, 3, 2).astype(np.float32)  # height=3, width=3, in_channels=3, out_channels=2
        conv_params = {
            'pads': (1, 1, 1, 1),  # top, bottom, left, right
            'strideh': 1,
            'stridew': 1,
            'dilations': (1, 1)
        }
        result = self.Conv2dOperation.conv2d(x, conv_filter, None, conv_params)
        assert "mock.compat.v1.Session().__enter__().run()" in str(result)
       