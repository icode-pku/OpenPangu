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

import os
import unittest

from model_evaluation.common import utils
from model_evaluation.common.enum import Framework


class TestOpMap(unittest.TestCase):

    def test_check_file_security(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        real_dir = os.path.join(cur_dir, 'testdir')
        os.mkdir(real_dir)
        self.assertFalse(utils.check_file_security(real_dir))
        os.removedirs(real_dir)
        self.assertTrue(utils.check_file_security(real_dir))

        real_file = os.path.join(cur_dir, 'test.sh')
        self.assertTrue(utils.check_file_security(real_file))

    def test_get_framework(self):
        model = 'xxx/test.onnx'
        framework = utils.get_framework(model)
        self.assertEqual(framework, Framework.ONNX)

        model = 'xxx/test.pb'
        framework = utils.get_framework(model)
        self.assertEqual(framework, Framework.TF)

        model = 'xxx/test.prototxt'
        framework = utils.get_framework(model)
        self.assertEqual(framework, Framework.CAFFE)

        model = 'xxx/test.txt'
        framework = utils.get_framework(model)
        self.assertEqual(framework, Framework.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
