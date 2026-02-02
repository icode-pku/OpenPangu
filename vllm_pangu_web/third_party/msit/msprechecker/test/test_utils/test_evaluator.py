# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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
from msprechecker.utils import Evaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator
    
    def test_addition(self):
        self.assertEqual(self.evaluator.evaluate("2 + 3"), 5)
        self.assertEqual(self.evaluator.evaluate("-2 + 3"), 1)
        self.assertEqual(self.evaluator.evaluate("-3 + 3"), 0)

    def test_subtraction(self):
        self.assertEqual(self.evaluator.evaluate("2 - 3"), -1)
        self.assertEqual(self.evaluator.evaluate("-2 - 3"), -5)
        self.assertEqual(self.evaluator.evaluate("3 - 3"), 0)
    
    def test_multiplication(self):
        self.assertEqual(self.evaluator.evaluate("2 * 3"), 6)
        self.assertEqual(self.evaluator.evaluate("-2 * -2"), 4)
        self.assertEqual(self.evaluator.evaluate("-3 * 3"), -9)
    
    def test_division(self):
        self.assertEqual(self.evaluator.evaluate("2 / 3"), 2 / 3)
        self.assertEqual(self.evaluator.evaluate("-2 / -2"), -2 / -2)
        self.assertEqual(self.evaluator.evaluate("-3 / 3"), -3 / 3)
    
    def test_equality(self):
        self.assertEqual(self.evaluator.evaluate("-2 + 3 == 3 - 2"), -2 + 3 == 3 - 2)
        self.assertEqual(self.evaluator.evaluate("-2 - 3 == -3 - 2"), -2 - 3 == -3 - 2)

    def test_inequality(self):
        self.assertEqual(self.evaluator.evaluate("-2 < 3"), -2 < 3)
        self.assertEqual(self.evaluator.evaluate("-1 > -2"), -1 > -2)
        self.assertEqual(self.evaluator.evaluate("3 <= 3"), 3 <= 3)
        self.assertEqual(self.evaluator.evaluate("5 != -5"), 5 != -5)
        self.assertEqual(self.evaluator.evaluate("5 >= 3"), 5 >= 3)
    
    def test_logical(self):
        self.assertEqual(self.evaluator.evaluate("1 == 2 or 2 != 2"), 1 == 2 or 2 != 2)
        self.assertEqual(
            self.evaluator.evaluate(
                "1 + 2 == 3 and not (1 + 2 == 3) or 1 // 2 == 0"), 
                1 + 2 == 3 and not (1 + 2 == 3) or 1 // 2 == 0
            )

    def test_plain_str(self):
        self.assertEqual(self.evaluator.evaluate("'afloata'"), "afloata")
        self.assertEqual(self.evaluator.evaluate("'ainta'"), "ainta")
        self.assertEqual(self.evaluator.evaluate("'aVersiona'"), "aVersiona")
        self.assertEqual(self.evaluator.evaluate("'strand'"), "strand")
        self.assertEqual(self.evaluator.evaluate("'sour'"), "sour")
        self.assertEqual(self.evaluator.evaluate("'knot'"), "knot")

    def test_function(self):
        self.assertEqual(
            self.evaluator.evaluate("str(123) == 123"), 
            str(123) == 123
        )
        self.assertEqual(
            self.evaluator.evaluate("int(123) == '123'"), 
            int(123) == '123'
        )
        self.assertEqual(
            self.evaluator.evaluate("int(str(123)) == 123"),
            int(str(123)) == 123
        )
        self.assertEqual(
            self.evaluator.evaluate("str('$(MINDIE_USER_HOME_PATH)/Ascend/mindie-service')"),
            "$(MINDIE_USER_HOME_PATH)/Ascend/mindie-service"
        )
        self.assertEqual(
            self.evaluator.evaluate("int('8') + int('8') // int('2')"), 
            12
        )
