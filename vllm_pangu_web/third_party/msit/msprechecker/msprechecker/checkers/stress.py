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

from statistics import mean

from .base import BaseChecker


class StressChecker(BaseChecker):
    def __init__(self, *, error_handler=None, rule_manager=None, threshold=0.2):
        super().__init__(error_handler=error_handler, rule_manager=rule_manager)
        self.error_handler.type = "stress"
        self.threshold = threshold

    def _check(self, results):
        if not isinstance(results, dict):
            raise TypeError(f"Expected 'collect_data.data' to be dict. Got {type(results).__name__} instead.")
        if not results:
            raise ValueError(f"No data collected and no errors occured during collection.")

        mean_time_cost = mean(results.values())
        if mean_time_cost == 0:
            mean_time_cost += 1e-6 # this should not happen since there will be no negative time cost

        expected_time_cost = self.threshold * mean_time_cost + mean_time_cost
        for i, time_cost in results.items():
            score = (time_cost - mean_time_cost) / mean_time_cost
            if score > self.threshold:
                self.error_handler.add_error(
                    path=f"Core ID: {i}",
                    actual=f"Time Cost: {time_cost:.3f}ms",
                    expected=f"Time Cost <= {expected_time_cost:.3f}ms",
                    reason=f"该核计算时长大于平均时长的 {score:.2%}",
                    severity="high" 
                )
