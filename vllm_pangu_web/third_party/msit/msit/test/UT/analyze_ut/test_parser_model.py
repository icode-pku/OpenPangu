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
from typing import List

from model_evaluation.common import utils
from model_evaluation.common.enum import AtcErr, Framework
from model_evaluation.bean import ConvertConfig
from model_evaluation.parser import ModelParser


class TestModelParser(unittest.TestCase):

    def setUp(self) -> None:
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.resource_dir = os.path.join(self.cur_dir, '..', 'resource', 'analyze')
        self.model_path = os.path.join(self.resource_dir, 'test.onnx')
        self.config = ConvertConfig(framework=Framework.ONNX)

        err_data_path = os.path.join(self.resource_dir, 'dataset', 'atc_err')
        err_file = 'atc_err_ez9010.txt'
        err_file_path = os.path.join(err_data_path, err_file)

        with open(err_file_path) as f:
            lines: List[str] = f.readlines()
            self.errinfo = ''.join(lines)

        if len(self.errinfo) == 0:
            self.errinfo = 'E19010: No parser is registered for Op [Conv_52, optype [ai.onnx::11::Convx]].'

    def test_parse_model_to_json_fail_case(self):
        model_parser = ModelParser(self.model_path, self.resource_dir, self.config)

        utils.exec_command = mock.Mock(return_value=('', 'Run error.'))
        res = model_parser.parse_model_to_json()
        self.assertFalse(res)

        utils.exec_command = mock.Mock(return_value=('Atc run failed', ''))
        res = model_parser.parse_model_to_json()
        self.assertFalse(res)

    def test_parse_model_to_json_success_case(self):
        model_parser = ModelParser(self.model_path, self.resource_dir, self.config)

        utils.exec_command = mock.Mock(return_value=('ATC run success', ''))
        res = model_parser.parse_model_to_json()
        self.assertTrue(res)

    def test_parse_model_to_om_fail_case(self):
        model_parser = ModelParser(self.model_path, self.resource_dir, self.config)

        utils.exec_command = mock.Mock(return_value=('', 'Run error.'))

        errcode, errinfo = model_parser.parse_model_to_om()
        self.assertEqual(errcode, AtcErr.UNKNOWN)
        self.assertEqual(errinfo, '')

    def test_parse_model_to_om_err_ez9010_case(self):
        model_parser = ModelParser(self.model_path, self.resource_dir, self.config)

        utils.exec_command = mock.Mock(return_value=(self.errinfo, ''))

        errcode, errinfo = model_parser.parse_model_to_om()
        self.assertEqual(errcode, AtcErr.EZ9010)
        self.assertNotEqual(len(errinfo), 0)

    def test_parse_all_ops_fail_case(self):
        model_parser = ModelParser(self.model_path, self.resource_dir, self.config)

        ops = model_parser.parse_all_ops()
        self.assertEqual(ops, [])

    def test_parse_all_ops_success_case(self):
        model_parser = ModelParser(self.model_path, self.resource_dir, self.config)

        src_path = os.path.join(self.resource_dir, 'dataset', 'model', 'test.onnx.json')
        dst_path = os.path.join(self.resource_dir, 'test.onnx.json')
        shutil.copyfile(src_path, dst_path)

        model_parser._json_path = dst_path
        op_infos = model_parser.parse_all_ops()
        self.assertNotEqual(op_infos, [])

        real_op_map = {'Sub_7': 'Sub', 'Div_9': 'Div'}
        for op_info in op_infos:
            self.assertEqual(op_info.op_type, real_op_map.get(op_info.op_name))


if __name__ == "__main__":
    unittest.main()
