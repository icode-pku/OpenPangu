# Copyright (c) 2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import os
import shutil
from unittest import mock
from typing import List, Set

import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto

from model_evaluation.bean import OpInfo
from model_evaluation.common import Const
from model_evaluation.common.enum import Framework, SocType, AtcErr
from model_evaluation.parser import ModelParser
from model_evaluation.bean import ConvertConfig
from model_evaluation.core import Analyze
from model_evaluation.core.result import OpResult


def make_new_onnx_model(onnx_path: str):
    weight = np.random.randn(9030)
    # Create input
    model_x0 = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 64600])
    conv_w = helper.make_tensor('W', TensorProto.FLOAT, [70, 1, 129], weight)
    # Create output
    model_out = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 70, 64472])

    # Create nodes: Conv, Abs
    node_conv = helper.make_node(
        'Conv', ['X', 'W'], ['O1'], 'Conv_0', dilations=[1], group=1, kernel_shape=[129], pads=[0, 0], strides=[1]
    )
    node_abs = helper.make_node('Abs', ['O1'], ['Y'], 'Abs_1')

    # Create the graph
    graph_def = helper.make_graph([node_conv, node_abs], 'test', [model_x0], [model_out], initializer=[conv_w])

    # Create the model
    model = helper.make_model(graph_def)

    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 11
    onnx.save(model, onnx_path)


class TestAnalyze(unittest.TestCase):
    def setUp(self) -> None:
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))
        resource_dir = os.path.join(self.cur_dir, '..', 'resource')
        self.onnx_model = os.path.join(self.cur_dir, 'test.onnx')
        make_new_onnx_model(self.onnx_model)

        self.ori_env_map = {
            'ASCEND_TOOLKIT_HOME': os.getenv('ASCEND_TOOLKIT_HOME'),
            'ASCEND_OPP_PATH': os.getenv('ASCEND_OPP_PATH'),
        }
        os.environ['ASCEND_TOOLKIT_HOME'] = ''
        os.environ['ASCEND_OPP_PATH'] = ''

        self.real_bin_path = os.path.dirname(Const.FAST_QUERY_BIN)
        if not os.path.exists(self.real_bin_path):
            os.makedirs(self.real_bin_path)
        mock_bin_path = os.path.join(resource_dir, 'analyze', 'mock', 'bin', 'ms_fast_query.py')
        shutil.copyfile(mock_bin_path, Const.FAST_QUERY_BIN)
        self.test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

    def tearDown(self) -> None:
        for env, val in self.ori_env_map.items():
            if val and len(val) != 0:
                os.environ[env] = val
            else:
                os.environ.pop(env)

        os.remove(Const.FAST_QUERY_BIN)
        os.removedirs(self.real_bin_path)

        os.remove(self.onnx_model)

    def test_update_result_with_err_op_types(self):
        model = self.onnx_model
        config = ConvertConfig(framework=Framework.ONNX, soc_type=SocType.Ascend310.name)
        analyze = Analyze(model, self.cur_dir, config)

        op_infos: List[OpInfo] = [
            OpInfo(op_name='Conv_52', op_type='Conv'),
            OpInfo(op_name='Conv_112', op_type='Conv'),
            OpInfo(op_name='Relu_34', op_type='Relu'),
        ]

        model_parser = ModelParser(model, self.cur_dir, config)
        model_parser.parse_all_ops = mock.Mock(return_value=op_infos)
        analyze._model_parser = model_parser

        analyze._result.insert(OpResult(ori_op_name='Conv_52', ori_op_type='Conv'))

        err_op_types: Set[str] = {'Conv'}
        analyze._update_result_with_err_op_types(err_op_types)

        for op_info in op_infos:
            if op_info.op_type not in err_op_types:
                continue
            op_result = analyze._result.get(op_info.op_name)
            self.assertIsNotNone(op_result)
            self.assertEqual(op_result.ori_op_type, op_info.op_type)
            self.assertFalse(op_result.is_supported)
            self.assertEqual(op_result.details, Const.ERR_UNSUPPORT)

    def test_update_result_with_err_ops(self):
        model = self.onnx_model
        config = ConvertConfig(framework=Framework.ONNX, soc_type=SocType.Ascend310.name)
        analyze = Analyze(model, self.cur_dir, config)

        err_ops = {'Conv_52': 'Conv', 'Conv_112': 'Conv', 'Relu_34': 'Relu'}
        analyze._result.insert(OpResult(ori_op_name='Conv_52', ori_op_type='Conv'))

        analyze._update_result_with_err_ops(err_ops)

        for ori_op_name, ori_op_type in err_ops.items():
            op_result = analyze._result.get(ori_op_name)
            self.assertIsNotNone(op_result)
            self.assertEqual(op_result.ori_op_type, ori_op_type)
            self.assertFalse(op_result.is_supported)
            self.assertEqual(op_result.details, Const.ERR_UNSUPPORT)

    def test_analyze_op_by_map_table_for_onnx(self):
        model = self.onnx_model
        config = ConvertConfig(framework=Framework.ONNX, soc_type=SocType.Ascend310.name)
        analyze = Analyze(model, self.cur_dir, config)

        ori_ops = [
            OpInfo(op_name='Abs_52', op_type='Abs'),
            OpInfo(op_name='Abs_112', op_type='Abs'),
            OpInfo(op_name='Relu_34', op_type='Relu'),
            OpInfo(op_name='MyOp_111', op_type='MyOp'),
        ]
        use_case = [
            ('Abs_52', 'Abs', True, ''),
            ('Abs_112', 'Abs', True, ''),
            ('Relu_34', 'Relu', True, ''),
            ('MyOp_111', 'MyOp', True, ''),
        ]

        os.environ['ASCEND_TOOLKIT_HOME'] = self.cur_dir
        os.environ['ASCEND_OPP_PATH'] = self.cur_dir

        analyze._init_result(ori_ops)
        analyze._analyze_op_by_map_table(ori_ops)

        for ori_op_name, ori_op_type, is_supported, err_detail in use_case:
            op_result = analyze._result.get(ori_op_name)
            self.assertIsNotNone(op_result)
            self.assertEqual(op_result.ori_op_type, ori_op_type)
            self.assertEqual(op_result.is_supported, is_supported)
            self.assertEqual(op_result.details, err_detail)

    def test_analyze_op_by_map_table_for_tf(self):
        model = os.path.join(self.cur_dir, 'test.pb')
        config = ConvertConfig(framework=Framework.TF, soc_type=SocType.Ascend310.name)
        analyze = Analyze(model, self.cur_dir, config)

        ori_ops = [
            OpInfo(op_name='Abs_52', op_type='Abs'),
            OpInfo(op_name='Abs_112', op_type='Abs'),
            OpInfo(op_name='Relu_34', op_type='Relu'),
            OpInfo(op_name='MyOp_111', op_type='MyOp'),
        ]
        use_case = [
            ('Abs_52', 'Abs', True, ''),
            ('Abs_112', 'Abs', True, ''),
            ('Relu_34', 'Relu', True, ''),
            ('MyOp_111', 'MyOp', True, ''),
        ]

        os.environ['ASCEND_TOOLKIT_HOME'] = self.cur_dir
        os.environ['ASCEND_OPP_PATH'] = self.cur_dir

        analyze._init_result(ori_ops)
        analyze._analyze_op_by_map_table(ori_ops)

        for ori_op_name, ori_op_type, is_supported, err_detail in use_case:
            op_result = analyze._result.get(ori_op_name)
            self.assertIsNotNone(op_result)
            self.assertEqual(op_result.ori_op_type, ori_op_type)
            self.assertEqual(op_result.is_supported, is_supported)
            self.assertEqual(op_result.details, err_detail)

    def test_check_constraint(self):
        model = self.onnx_model
        config = ConvertConfig(framework=Framework.ONNX, soc_type=SocType.Ascend310.name)
        analyze = Analyze(model, self.cur_dir, config)

        graph = analyze._graph.graph
        for node in graph.node:
            analyze._result.insert(OpResult(ori_op_name=node.name, ori_op_type=node.op_type))
        analyze._check_op_constraint()

        for op_result in analyze._result._op_results.values():
            if op_result.is_supported:
                self.assertTrue(op_result.is_supported)
                self.assertEqual(op_result.details, '')
            else:
                self.assertFalse(op_result.is_supported)
                self.assertNotEqual(op_result.details, '')

    def test_analyze_model(self):
        model = self.onnx_model
        config = ConvertConfig(framework=Framework.ONNX, soc_type=SocType.Ascend310.name)
        analyze = Analyze(model, self.cur_dir, config)

        ori_ops = [
            OpInfo(op_name='Abs_52', op_type='Abs'),
            OpInfo(op_name='Abs_112', op_type='Abs'),
            OpInfo(op_name='Relu_34', op_type='Relu'),
            OpInfo(op_name='MyOp_111', op_type='MyOp'),
        ]
        use_case = [
            ('Abs_52', 'Abs', True, ''),
            ('Abs_112', 'Abs', True, ''),
            ('Relu_34', 'Relu', True, ''),
            ('MyOp_111', 'MyOp', False, Const.ERR_UNSUPPORT),
        ]

        os.environ['ASCEND_TOOLKIT_HOME'] = self.cur_dir
        os.environ['ASCEND_OPP_PATH'] = self.cur_dir

        errinfo = (
            'ATC start working now, please wait for a moment.'
            'ATC run failed, Please check the detail log, Try \'atc --help\' for more information'
            'EZ3003: No supported Ops kernel and engine are found for [MyOp_111], optype [MyOp].'
        )

        analyze._model_parser = ModelParser(model, self.cur_dir, config)
        analyze._model_parser.parse_all_ops = mock.Mock(return_value=ori_ops)
        analyze._model_parser.parse_model_to_om = mock.Mock(return_value=(AtcErr.EZ3003, errinfo))
        analyze._check_op_constraint = mock.Mock()

        analyze.analyze_model()

        for ori_op_name, ori_op_type, is_supported, err_detail in use_case:
            op_result = analyze._result.get(ori_op_name)
            self.assertIsNotNone(op_result)
            self.assertEqual(op_result.ori_op_type, ori_op_type)
            self.assertEqual(op_result.is_supported, is_supported)
            self.assertEqual(op_result.details, err_detail)
        os.remove(os.path.join(self.cur_dir, "result.csv"))


if __name__ == '__main__':
    unittest.main()
