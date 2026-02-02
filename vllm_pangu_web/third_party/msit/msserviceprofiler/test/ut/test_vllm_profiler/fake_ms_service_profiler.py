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

from collections import namedtuple


# Revised ProfilerMock that properly tracks instance calls
class Profiler:
    instance_calls = []

    def __init__(self, level=None):
        self.calls = []
        Profiler.instance_calls.append(self.calls)

    @classmethod
    def reset(cls):
        cls.instance_calls = []

    def domain(self, name):
        self.calls.append(("domain", name))
        return self

    def res(self, res_id):
        self.calls.append(("res", res_id))
        return self

    def event(self, event_name):
        self.calls.append(("event", event_name))
        return self

    def metric(self, name, value):
        self.calls.append(("metric", name, value))
        return self

    def metric_scope(self, name, value):
        self.calls.append(("metric_scope", name, value))
        return self

    def metric_inc(self, name, value):
        self.calls.append(("metric_inc", name, value))
        return self

    def span_start(self, name):
        self.calls.append(("span_start", name))
        return self

    def span_end(self):
        self.calls.append("span_end")
        return self

    def attr(self, name, value):
        self.calls.append(("attr", name, value))
        return self


# Create fake modules in sys.modules
Level = namedtuple("Level", ["INFO"])("INFO")
