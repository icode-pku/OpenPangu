# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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


from unittest import TestCase

from msit_llm.metrics.metrics import Accuracy, EditDistance, get_metric


class TestMetrics(TestCase):

    def test_accuracy_score_positive(self):
        outs = ["1"]
        refs = ["1"]
        thr = 0

        self.accuracy = Accuracy(thr)
        generator = self.accuracy.compare_two_lists_of_words(outs, refs)

        with self.assertRaises(StopIteration):
            next(generator)

    def test_accuracy_score_negative(self):
        outs = ["1"]
        refs = [""]
        thr = 1
        
        self.accuracy = Accuracy(thr)
        generator = self.accuracy.compare_two_lists_of_words(outs, refs)

        self.assertEqual(next(generator), (0, 0))

    def test_edit_distance_positive(self):
        outs = ["a"]
        refs = ["a"]

        edit_distance = EditDistance(0)
        generator = edit_distance.compare_two_lists_of_words(outs, refs)

        with self.assertRaises(StopIteration):
            next(generator)
    
    def test_edit_distance_diff_thr(self):
        outs = ["a"] * 1000000
        refs = [" "] * 1000000

        edit_distance = EditDistance(0)
        generator = edit_distance.compare_two_lists_of_words(outs, refs)
        self.assertEqual(next(generator), (0, 1))

        edit_distance = EditDistance(2)
        generator = edit_distance.compare_two_lists_of_words(outs, refs)
        with self.assertRaises(StopIteration):
            next(generator)

    def test_get_metrics_should_raise_when_name_not_valid(self):
        with self.assertRaises(TypeError):
            get_metric(1)
        
        with self.assertRaises(TypeError):
            get_metric(1.5)
        
        with self.assertRaises(TypeError):
            get_metric([])

    def test_get_metrics_should_raise_when_name_not_support(self):
        with self.assertRaises(KeyError):
            get_metric("")

        with self.assertRaises(KeyError):
            get_metric("acc")        

    def test_get_metrics_should_raise_when_thr_not_valid(self):
        with self.assertRaises(TypeError):
            get_metric("accuracy", "")
