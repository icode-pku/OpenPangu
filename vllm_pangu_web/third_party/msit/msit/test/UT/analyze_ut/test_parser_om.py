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

from unittest.mock import patch

from model_evaluation.bean import OpInnerInfo
from model_evaluation.common import utils
from model_evaluation.common.enum import Engine
from model_evaluation.parser import OmParser


class TestOmParser(unittest.TestCase):

    def setUp(self) -> None:
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.om_path = os.path.join(self.cur_dir, 'test.om')
        self.resource_path = os.path.join(self.cur_dir, '..', 'resource', 'analyze')

    def test_parse_om_to_json_success_case(self):
        om_parser = OmParser(self.om_path, self.cur_dir)
        with patch('model_evaluation.common.utils.exec_command') as mock_exec_command:
            mock_exec_command.return_value = ('ATC run success', '')
            res = om_parser.parse_om_to_json()
            self.assertTrue(res)

    def test_parse_om_to_json_fail_case(self):
        om_parser = OmParser(self.om_path, self.cur_dir)
        with patch('model_evaluation.common.utils.exec_command') as mock_exec_command:
            mock_exec_command.return_value = ('', 'Run error.')
            res = om_parser.parse_om_to_json()
            self.assertFalse(res)

    def test_parse_all_ops_fail_case(self):
        om_parser = OmParser(self.om_path, self.cur_dir)

        op_infos = om_parser.parse_all_ops()
        self.assertEqual(op_infos, [])

    def test_parse_all_ops_success_case(self):
        om_parser = OmParser(self.om_path, self.cur_dir)

        src_path = os.path.join(self.resource_path, 'dataset', 'model', 'test.om.json')
        dst_path = os.path.join(self.cur_dir, 'test.om.json')
        shutil.copyfile(src_path, dst_path)

        om_parser._om_json = dst_path
        op_infos = om_parser.parse_all_ops()
        self.assertNotEqual(op_infos, [])

        real_op_map = {
            'LeakyRelu_54': OpInnerInfo(
                op_name='LeakyRelu_54', op_type='LeakyRelu', op_engine=Engine.AICORE, is_fusion=False
            ),
            'PartitionedCall_Conv_112_Conv2D_32': OpInnerInfo(
                op_name='PartitionedCall_Conv_112_Conv2D_32', op_type='Conv2D', op_engine=Engine.AICORE, is_fusion=False
            ),
            'Div_331Mul_333Add_335Mul_337Add_338': OpInnerInfo(
                op_name='Div_331Mul_333Add_335Mul_337Add_338',
                op_type='RealDiv',
                op_engine=Engine.AICORE,
                is_fusion=True,
            ),
        }
        op = real_op_map.get('Div_331Mul_333Add_335Mul_337Add_338')
        op.ori_ops = ["Div_331", "Mul_333", "Add_335", "Mul_337", "Add_338"]

        hit_op = set()
        for op_info in op_infos:
            if op_info.op_name not in real_op_map:
                continue
            hit_op.add(op_info.op_name)
            real_op_info = real_op_map.get(op_info.op_name)
            self.assertEqual(op_info.op_name, real_op_info.op_name)
            self.assertEqual(op_info.op_type, real_op_info.op_type)
            self.assertEqual(op_info.op_engine, real_op_info.op_engine)
            self.assertEqual(op_info.is_fusion, real_op_info.is_fusion)

            if real_op_info.is_fusion:
                self.assertEqual(op_info.ori_ops, real_op_info.ori_ops)

        for real_op_name in real_op_map.keys():
            self.assertIn(real_op_name, hit_op)

        os.remove(dst_path)


if __name__ == "__main__":
    unittest.main()
