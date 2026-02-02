import unittest
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal


class TestGoldenFuncsOperation(unittest.TestCase):

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
        from msit_opcheck.golden_funcs import OP_DICT
        from msit_opcheck.operation_test import OperationTest
        self.OperationTest = OperationTest
        self.OP_DICT = OP_DICT
        
    def tearDown(self):
        for mod in self.modules_to_mock:
            if mod in self.backup_modules:
                sys.modules[mod] = self.backup_modules[mod]
            else:
                del sys.modules[mod] 
        for mod in list(sys.modules.keys()):
            if mod.startswith("msit_opcheck.golden_funcs"):
                del sys.modules[mod]

    def test_add_operation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            inputs = [np.array([1, 2, 3]), np.array([4, 5, 6])]
            add_op = self.OP_DICT["Add"]()
            add_op.op_param = {
                'output_desc': [{'dtype': 'DT_FLOAT'}]
            }
            result = add_op.golden_calc(inputs)[0]
            expected = np.array([5.0, 7.0, 9.0])
            assert_equal(result, expected)

    def test_BatchMatMulOperation_with_bias(self):
        x1 = np.random.rand(2, 3, 4).astype(np.float32)
        x2 = np.random.rand(2, 4, 5).astype(np.float32)
        bias = np.random.rand(5).astype(np.float32)
        with patch.object(self.OperationTest, "__init__", return_value=None):
            matmul_op = self.OP_DICT["BatchMatMulV2"]()
            matmul_op.op_param = {
                'output_desc': [{'dtype': 'DT_FLOAT'}],
                'attr': [{'key': 'adj_x1', 'value': {'b': False}},
                         {'key': 'adj_x2', 'value': {'b': False}}]
            }
            result = matmul_op.golden_calc([x1, x2, bias])
            self.assertEqual(result[0].shape, (2, 3, 5))
    
    def test_BatchNormOperation(self):
        inputs = (
            np.random.rand(1, 3, 224, 224).astype(np.float32),
            np.random.rand(3).astype(np.float64),
            np.random.rand(3).astype(np.float64),
            np.random.rand(3).astype(np.float64),
            np.random.rand(3).astype(np.float64)
        )
        with patch.object(self.OperationTest, "__init__", return_value=None):
            batchnorm_op = self.OP_DICT["BatchNorm"]()
            batchnorm_op.op_param = {
                'input_desc': [
                    {'dtype': 'DT_FLOAT'},
                    {'dtype': 'DT_DOUBLE'}
                ],
                'output_desc': [
                    {'dtype': 'DT_FLOAT'}
                ],
                'attr': [
                    {'key': 'epsilon', 'value': {'f': 1e-5}}
                ]
            }
            result = batchnorm_op.golden_calc(inputs)
            self.assertEqual(result[0].dtype, np.float64)
            # test output: float16 or bfloat16
            batchnorm_op.op_param['output_desc'][0]['dtype'] = 'DT_FLOAT16'
            result = batchnorm_op.golden_calc(inputs)
            self.assertEqual(result[0].dtype, np.float64)

    def test_BiasAddOperation(self):
        value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        bias = np.array([1, 2])
        expected = np.array([[[2, 4], [4, 6]], [[6, 8], [8, 10]]])
        with patch.object(self.OperationTest, "__init__", return_value=None):
            bias_add_op = self.OP_DICT["BiasAdd"]()
            result = bias_add_op.golden_calc([value, bias])[0]
            assert_array_equal(result, expected)

    def test_BninferenceOperation(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mean = np.array([0.5], dtype=np.float32)
        variance = np.array([0.25], dtype=np.float32)
        scale = np.array([1.5], dtype=np.float32)
        bias = np.array([0.1], dtype=np.float32)
        with patch.object(self.OperationTest, "__init__", return_value=None):
            bninfer_op = self.OP_DICT["BNInfer"]()
            bninfer_op.op_param = {
                'input_desc': [{'layout': 'NCHW'}]
            }
            # test_basic_case_no_scale_no_bias
            result = bninfer_op.golden_calc([x, mean, variance])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].dtype, np.float32)

            # test_with_scale_no_bias
            result = bninfer_op.golden_calc([x, mean, variance, scale])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].dtype, np.float32)

            # test_with_scale_and_bias
            result = bninfer_op.golden_calc([x, mean, variance, scale, bias])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].dtype, np.float32)

    def test_CastOperation_golden_calc(self):
        input_data = np.array([1.5, 2.3, 3.7], dtype=np.float32)
        uint1_input_data = np.array([1, 0, 1], dtype=np.uint8)
        expected_int32 = input_data.astype(np.int32)
        expected_int64 = input_data.astype(np.int64)
        expected_float16 = input_data.astype(np.float16)
        with patch.object(self.OperationTest, "__init__", return_value=None):
            cast_op = self.OP_DICT["Cast"]()
            cast_op.op_param = {
                'input_desc': [{'dtype': 'DT_FLOAT'}],
                'output_desc': [{'dtype': 'DT_INT64'}]
            }
            result = cast_op.golden_calc([input_data])[0]
            assert_array_equal(result, expected_int64)

            # test output: complex32
            cast_op.op_param['output_desc'][0]['dtype'] = 'DT_COMPLEX32'
            result = cast_op.golden_calc([input_data])[0]
            expected_complex32_shape = list(input_data.shape) + [2]
            self.assertEqual(result.shape, tuple(expected_complex32_shape))
            np.testing.assert_array_equal(result[..., 0], input_data)
            np.testing.assert_array_equal(result[..., 1], np.zeros_like(input_data))

            # test src uint1
            cast_op.op_param['input_desc'][0]['dtype'] = 'DT_UINT1'
            result = cast_op.golden_calc([uint1_input_data])[0]

            # test float32 to float16
            cast_op.op_param['input_desc'][0]['dtype'] = 'DT_FLOAT'
            cast_op.op_param['output_desc'][0]['dtype'] = 'DT_FLOAT16'
            result = cast_op.golden_calc([input_data])[0]
            np.testing.assert_array_equal(result, expected_float16)

            # test output: complex64
            cast_op.op_param['output_desc'][0]['dtype'] = 'DT_COMPLEX64'
            result = cast_op.golden_calc([input_data])[0]

            # test general
            cast_op.op_param['output_desc'][0]['dtype'] = 'DT_INT32'
            result = cast_op.golden_calc([input_data])[0]
            np.testing.assert_array_equal(result, expected_int32)

    def test_CastOperation_numpy_to_torch_tensor(self):
        np_array = np.array([1.0, 2.0], dtype=np.float16)
        with patch.object(self.OperationTest, "__init__", return_value=None):
            cast_op = self.OP_DICT["Cast"]()
            # complex32_conversion
            torch_tensor = cast_op.numpy_to_torch_tensor(np_array, is_complex32=True)
            self.assertEqual(torch_tensor.dtype, torch.complex32)
            self.assertEqual(torch_tensor.shape, (1,))

    def test_ClipByValueOperation(self):
        input_t = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        clip_min = np.array([2.0], dtype=np.float32)
        clip_max = np.array([3.0], dtype=np.float32)
        expected = np.array([2.0, 2.0, 3.0, 3.0], dtype=np.float32)
        with patch.object(self.OperationTest, "__init__", return_value=None):
            clip_by_value_op = self.OP_DICT["ClipByValue"]()
            result = clip_by_value_op.golden_calc([input_t, clip_min, clip_max])[0]
            assert_array_equal(result, expected)
    
    def test_ConcatOperation(self):
        basic_in_tensors = [
            np.array([[[[1, 2], [3, 4]]]]),  # 1x1x2x2
            np.array([[[[5, 6], [7, 8]]]])   # 1x1x2x2
        ]
        nchw_in_tensors = [
            np.array([[[[1], [2]], [[3], [4]]]]),  # NHWC format
            np.array([[[[5], [6]], [[7], [8]]]])
        ]
        basic_expected = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
        with patch.object(self.OperationTest, "__init__", return_value=None):
            concat_op = self.OP_DICT["ConcatV2"]()
            concat_op.op_param = {
                'input_desc': [{
                    'layout': 'NCHW',
                    'attr': []
                }],
                'attr': []
            }
            # test basic
            concat_op.op_param['attr'] = [{'key': 'concat_dim', 'value': {'i': 1}}]
            result = concat_op.golden_calc(basic_in_tensors)
            assert_array_equal(result[0], basic_expected)

            # test concat_with_origin_format_conversion
            concat_op.op_param['input_desc'][0]['attr'] = [
                {'key': 'origin_format', 'value': {'s': 'NHWC'}},
                {'key': 'origin_shape', 'value': {'list': {'i': [1, 2, 2, 1]}}}
            ]
            result = concat_op.golden_calc(nchw_in_tensors)
            self.assertEqual(result[0].shape, (1, 4, 2, 1))

    def test_ConcatV2Operation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None), \
             patch("msit_opcheck.golden_funcs.conv2d.Conv2dOperation.conv2d", return_value=[1,2,3]), \
             patch("msit_opcheck.golden_funcs.conv2d.Conv2dOperation.conv2d_with_groups", return_value=[4,5]):
            concat_v2d_op = self.OP_DICT["Conv2D"]()
            concat_v2d_op.get_conv_params = lambda: ({"pads": [0, 1, 0, 1], "strideh":1, "stridew":1, "dilations":0}, 1)
            # test basic conv2d
            x = np.ones((1, 3, 3, 1), dtype=np.float32)
            conv_filter = np.ones((1, 2, 2, 1), dtype=np.float32)
            result = concat_v2d_op.golden_calc([x, conv_filter])
            self.assertEqual(result[0], [1, 2, 3])
            # test conv2d_with_groups
            concat_v2d_op.get_conv_params = lambda: ({}, 2)
            result = concat_v2d_op.golden_calc([x, conv_filter])
            self.assertEqual(result[0], [4, 5])

    def test_GatherOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            gather_op = self.OP_DICT["GatherV2"]()
            gather_op.op_param = {
                'attr': [
                    {'key': 'axis', 'value': {'i': 1}},
                    {'key': 'batch_dims', 'value': {'i': 0}}
                ]
            }
            params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
            indices = np.array([0, 1], dtype=np.int32)
            result = gather_op.golden_calc([params, indices])
            assert "mock.compat.v1.Session().__enter__().run()" in str(result)

    def test_LessOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            less_op = self.OP_DICT["Less"]()
            # 相同形状向量
            x1 = np.array([1, 2, 3])
            x2 = np.array([2, 1, 3])
            res = less_op.golden_calc([x1, x2])
            assert "mock.compat.v1.Session().__enter__().run()" in str(res)
            # 广播比较
            x1 = np.array([[1], [2], [3]])
            x2 = np.array([2])
            res2 = less_op.golden_calc([x1, x2])
            assert "mock.compat.v1.Session().__enter__().run()" in str(res2)

    def test_LogicalAndOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            logical_and_op = self.OP_DICT["LogicalAnd"]()
            logical_and_op.op_param = {'output_desc': [{'dtype': 'DT_BOOL'}]}
            x1 = np.array([True, False, True], dtype=bool)
            x2 = np.array([True, True, False], dtype=bool)
            result = logical_and_op.golden_calc([x1, x2])[0]
            expected = np.array([True, False, False], dtype=bool)
            assert_array_equal(result, expected)

    def test_LogicalOrOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            logical_or_op = self.OP_DICT["LogicalOr"]()
            logical_or_op.op_param = {'output_desc': [{'dtype': 'DT_BOOL'}]}
            x1 = np.array([True, False, True], dtype=bool)
            x2 = np.array([True, True, False], dtype=bool)
            result = logical_or_op.golden_calc([x1, x2])[0]
            expected = np.array([True, True, True], dtype=bool)
            assert_array_equal(result, expected)

    def test_MatMulOperation_gloden_calc(self):
        with patch.object(self.OperationTest, "__init__", return_value=None), \
             patch("msit_opcheck.golden_funcs.mat_mul.matmul", return_value="mock_matmul"):
            matmul_op = self.OP_DICT["MatMulV2"]()
            matmul_op.op_param = {
                'output_desc': [{'dtype': 'DT_FLOAT'}],
                'attr': [
                    {'key': 'transpose_x1', 'value': {'b': True}},
                    {'key': 'transpose_x2', 'value': {'b': True}}
                ],
                'input_desc': [
                    {'layout': 'NCHW'}, {'layout': 'NCHW'}, 
                    {'layout': 'NC1HWC0', 'attr': [
                        {'key': 'origin_format', 'value': {'s': 'NCHW'}},
                        {'key': 'origin_shape', 'value': {'list': {'i': [1, 16, 2, 2]}}}
                    ]}
                ]
            }
            x1 = np.random.rand(3, 2).astype(np.float32)
            x2 = np.random.rand(4, 3).astype(np.float32)
            bias = np.random.rand(1, 1, 2, 2, 16).astype(np.float32)
            result = matmul_op.golden_calc([x1, x2, bias])
            self.assertEqual(result[0], "mock_matmul")

    def test_MatMulOperation_hf_32_input_gerenate(self):
        from msit_opcheck.golden_funcs.mat_mul import hf_32_input_gerenate
        input_fp32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        output = hf_32_input_gerenate(input_fp32)
        expected = np.array([1.00048828, 2.00097656, 3.00146484], dtype=np.float32)
        assert np.allclose(output, expected, rtol=1e-3)

    def test_MinimumOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            minimum_op = self.OP_DICT["Minimum"]()
            minimum_op.op_param = {
                'output_desc': [{'dtype': 'DT_FLOAT'}]
            }
            x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            x2 = np.array([3.0, 2.0, 1.0], dtype=np.float32)
            result = minimum_op.golden_calc([x1, x2])[0]
            expected = np.array([1.0, 2.0, 1.0], dtype=np.float32)
            assert_array_equal(result, expected)

    def test_MulOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            mul_op = self.OP_DICT["Mul"]()
            mul_op.op_param = {
                'output_desc': [{'dtype': 'DT_FLOAT'}],
                'input_desc': [
                    {'dtype': 'DT_FLOAT'},
                    {'dtype': 'DT_FLOAT'}
                ]
            }
            # 测试普通浮点数乘法
            x1 = np.array([1.0, 2.0, 3.0])
            x2 = np.array([4.0, 5.0, 6.0])
            result = mul_op.golden_calc([x1, x2])[0]
            assert "mock.compat.v1.compat.v1.Session().__enter__().run().astype()" in str(result)
            # 测试复数乘法
            mul_op.op_param['input_desc'][0]['dtype'] = 'DT_COMPLEX32'
            mul_op.op_param['input_desc'][1]['dtype'] = 'DT_COMPLEX32'
            x1 = np.array([1.0, 2.0, 3.0, 4.0])  # 表示1+2j和3+4j
            x2 = np.array([5.0, 6.0, 7.0, 8.0])  # 表示5+6j和7+8j
            result = mul_op.golden_calc([x1, x2])[0]
            assert_array_equal(result, np.array([-16.0, -20.0, 22.0, 40.0]))
            # 测试布尔类型乘法
            mul_op.op_param['input_desc'][0]['dtype'] = 'DT_BOOL'
            mul_op.op_param['input_desc'][1]['dtype'] = 'DT_BOOL'
            x1 = np.array([True, False, True])
            x2 = np.array([True, True, False])
            result = mul_op.golden_calc([x1, x2])[0]
            assert_array_equal(result, np.array([True, False, False]))

    def test_PackOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            pack_op = self.OP_DICT["Pack"]()
            pack_op.op_param = {
                'input_desc': [{
                    'layout': 'NHWC',
                    'attr': [
                        {'key': 'origin_format', 'value': {'s': 'NCHW'}},
                        {'key': 'origin_shape', 'value': {'list': {'i': [2, 3, 4, 5]}}}
                    ]
                }],
                'attr': [{'key': 'axis', 'value': {'i': 1}}]
            }
            in_tensors = [np.random.rand(2, 4, 5, 3), np.random.rand(2, 4, 5, 3)]
            result = pack_op.golden_calc(in_tensors)
            expected = np.concatenate(in_tensors, axis=1)
            assert_array_equal(result[0], expected)

    def test_PadOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            pad_op = self.OP_DICT["Pad"]()
            pad_op.op_param = {
                'input_desc': [
                        {
                            'attr': [
                                {'key': 'origin_format', 'value': {'s': 'NC1HWC0'}}
                            ]
                        }
                    ]
            }
            input_data = np.random.rand(2, 3, 4, 5, 6)  # NC1HWC0格式
            paddings = [[1, 1], [2, 2], [3, 3]]  # 对应H,W,C0的padding
            result = pad_op.golden_calc([input_data, paddings])
            self.assertEqual(result[0].shape, (2, 5, 8, 11, 6))
    
    def test_ReduceSumOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            reduce_sum_op = self.OP_DICT["ReduceSum"]()
            reduce_sum_op.op_param = {
                'input_desc': [{'dtype': "DT_FLOAT"}],
                'output_desc': [{'dtype': "DT_FLOAT"}],
                'attr': []
            }
            input_tensor = np.array([1, 2, 3])
            reduce_sum_op.op_param['attr'] = []
            result = reduce_sum_op.golden_calc([input_tensor])
            np.testing.assert_array_equal(result[0], input_tensor)

    def test_ReduceMeanOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            reduce_mean_op = self.OP_DICT["ReduceMean"]()
            # 测试没有指定axis时的场景
            in_tensors = [np.array([1, 2, 3, 4], dtype=np.float32)]
            reduce_mean_op.op_param = {
                'input_desc': [{'dtype': 'DT_FLOAT'}],
                'output_desc': [{'dtype': 'DT_FLOAT'}],
                'attr': []
            }
            result = reduce_mean_op.golden_calc(in_tensors)
            assert_array_equal(result[0], in_tensors[0])
            # 测试float16类型的输入
            reduce_mean_op.op_param = {
                'input_desc': [{'dtype': 'DT_FLOAT16'}],
                'output_desc': [{'dtype': 'DT_FLOAT16'}],
                'attr': [{'key': "axes", 'value': {'list': {'i': [0, 1]}}}]
            }
            in_tensors = [np.array([[1, 2], [3, 4]], dtype=np.float16)]
            result = reduce_mean_op.golden_calc(in_tensors)
            expected = np.sum(in_tensors[0].astype(np.float32) / 4, axis=(0, 1)).astype(np.float16)
            assert_array_equal(result[0], expected)

    def test_ReluOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            relu_op = self.OP_DICT["Relu"]()
            input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            result = relu_op.golden_calc([input_data])[0]
            assert_array_equal(result, expected)

    def test_RsqrtOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            rsqrt_op = self.OP_DICT["Rsqrt"]()
            rsqrt_op.op_param = {
                'output_desc': [{'dtype': 'DT_FLOAT'}]
            }
            input_tensor = np.array([4.0, 16.0, 64.0])
            expected_output = 1 / np.sqrt(input_tensor)
            result = rsqrt_op.golden_calc([input_tensor])[0]
            assert_array_almost_equal(result, expected_output)

    def test_SelectOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            select_op = self.OP_DICT["Select"]()
            select_op.op_param = {
                'output_desc': [{'dtype': 'DT_FLOAT'}]
            }
            condition = np.array([True, False, True])
            x1 = np.array([1, 2, 3], dtype=np.float32)
            x2 = np.array([4, 5, 6], dtype=np.float32)
            result = select_op.golden_calc([condition, x1, x2])[0]
            expected = np.array([1, 5, 3], dtype=np.float32)
            np.testing.assert_array_equal(result, expected)

    def test_SigmoidOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            sigmoid_op = self.OP_DICT["Sigmoid"]()
            sigmoid_op.op_param = {
                'output_desc': [{'dtype': 'DT_FLOAT16'}]
            }
            input_data = np.array([-1.5, 0.0, 1.5], dtype=np.float32)
            result = sigmoid_op.golden_calc([input_data])
            self.assertEqual(result[0].dtype, np.float16)

    def test_SoftmaxOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            softmax_op = self.OP_DICT["SoftmaxV2"]()
            softmax_op.op_param = {
                'attr': [{'key': 'axes', 'value': {'list': {'i': [1]}}}],
                'input_desc': [{
                    'dtype': 'DT_FLOAT',
                    'attr': [{'key': 'origin_shape', 'value': {'list': {'i': [2, 3]}}}]
                }],
                'output_desc': [{'dtype': 'DT_FLOAT'}]
            }
            softmax_op.op_param['input_desc'][0]['dtype'] = 'DT_FLOAT16'
            softmax_op.op_param['output_desc'][0]['dtype'] = 'DT_FLOAT'
            input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float16)
            result = softmax_op.golden_calc([input_data])[0]
            self.assertEqual(result.dtype, np.float32)

    def test_StridedSliceOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            stridedslice_op = self.OP_DICT["StridedSlice"]()
            stridedslice_op.op_param = {
                'attr': [
                    {'key': 'begin', 'value': {'list': {'i': [0, 1]}}},
                    {'key': 'end', 'value': {'list': {'i': [2, 3]}}},
                    {'key': 'strides', 'value': {'list': {'i': [1, 1]}}},
                    {'key': 'begin_mask', 'value': {'i': 0}},
                    {'key': 'end_mask', 'value': {'i': 0}},
                    {'key': 'ellipsis_mask', 'value': {'i': 0}},
                    {'key': 'new_axis_mask', 'value': {'i': 0}},
                    {'key': 'shrink_axis_mask', 'value': {'i': 0}}
                ]
            }
            x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            result = stridedslice_op.golden_calc([x])[0]
            assert "mock.compat.v1.Session().__enter__().run()" in str(result)

    def test_SubOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            sub_op = self.OP_DICT["Sub"]()
            # 浮点数减法
            x1 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
            x2 = np.array([0.5, 1.5, 2.5], dtype=np.float32)
            result = sub_op.golden_calc([x1, x2])[0]
            assert "mock.compat.v1.Session().__enter__().run()" in str(result)
            # 布尔值减法
            x1 = np.array([True, True, False, False], dtype=bool)
            x2 = np.array([True, False, True, False], dtype=bool)
            result = sub_op.golden_calc([x1, x2])[0]
            expected = np.logical_xor(x1, x2)
            assert_array_equal(result, expected)

    def test_TanhOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            tanh_op = self.OP_DICT["Tanh"]()
            tanh_op.op_param = {
                'output_desc': [{'dtype': 'DT_FLOAT16'}]
            }
            fp16_input_data = np.array([-0.5, 0.0, 0.5], dtype=np.float16)
            fp32_input_data = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
            # 测试fp16
            expected = np.tanh(fp16_input_data.astype(np.float32)).astype(np.float16)
            result = tanh_op.golden_calc([fp16_input_data])[0]
            assert_array_almost_equal(result, expected, decimal=3)
            # 测试fp32
            result = tanh_op.golden_calc([fp32_input_data])[0]
            assert_array_almost_equal(result, np.tanh(fp32_input_data), decimal=3)

    def test_TileDOperation(self):
        input_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        with patch.object(self.OperationTest, "__init__", return_value=None):
            tiled_op = self.OP_DICT["Tile"]()
            # 通过attr属性传入multiples
            tiled_op.op_param = {
                'attr': [{'key': 'multiples', 'value': {'list': {'i': [2, 3]}}}],
            }
            result = tiled_op.golden_calc([input_data])[0]
            assert "mock.compat.v1.Session().__enter__().run()" in str(result)
            # 通过输入张量传入multiples
            multiples = np.array([1, 1], dtype=np.int32)
            result = tiled_op.golden_calc([input_data, multiples])[0]
            assert "mock.compat.v1.Session().__enter__().run()" in str(result)

    def test_TransposeOperation(self):
        with patch.object(self.OperationTest, "__init__", return_value=None):
            transpose_op = self.OP_DICT["Transpose"]()
            input_0 = np.array([[1, 2], [3, 4]])
            perm = (1, 0)
            result = transpose_op.golden_calc([input_0, perm])[0]
            assert_array_equal(result, np.transpose(input_0, perm))
    