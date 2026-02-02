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
from components.expert_load_balancing.elb.algorithm_runner.base_algorithm_runner import AlgorithmType
from components.expert_load_balancing.elb.algorithm_runner.speculative_moe_runner import SpeculativeMoeRunner
from components.expert_load_balancing.elb.algorithm_runner.c2lb_runner import StaticC2lbRunner, StaticC2lbA3Runner, \
    DynamicC2lbRunner
from components.expert_load_balancing.elb.constant import A2, A3


class AlgorithmFactory:
    SPECULATIVE_MOE_RUNNER = [AlgorithmType.SPECULATIVE_MOE_LEVEL_1, AlgorithmType.SPECULATIVE_MOE_LEVEL_1_MIXED,
                              AlgorithmType.SPECULATIVE_MOE_LEVEL_2, AlgorithmType.SPECULATIVE_MOE_LEVEL_2_MIXED]

    @staticmethod
    def create_runner(args):
        def get_runner_type(algorithm_type, device_type):
            if algorithm_type in AlgorithmFactory.SPECULATIVE_MOE_RUNNER:
                return SpeculativeMoeRunner
            if algorithm_type == AlgorithmType.C2LB and device_type == A2:
                return StaticC2lbRunner
            if algorithm_type == AlgorithmType.C2LB and device_type == A3:
                return StaticC2lbA3Runner
            if algorithm_type == AlgorithmType.DYNAMIC_C2LB:
                return DynamicC2lbRunner
            raise ValueError("Cannot find algorithm runner for input options.")

        args.algorithm = AlgorithmType(int(args.algorithm))
        runn_type = get_runner_type(args.algorithm, args.device_type)
        return runn_type(args)


def run_algorithm(data, args):
    algorithm_runner = AlgorithmFactory.create_runner(args)
    algorithm_runner.run_algorithm(data)
