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
from components.expert_load_balancing.elb.data_loader.data_loader_factory import load_data
from components.expert_load_balancing.elb.algorithm_runner.algorithm_fatcory import run_algorithm


def load_balancing(args):
    data, new_args = load_data(args)
    run_algorithm(data, new_args)
