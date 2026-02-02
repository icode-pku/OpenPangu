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
import os

import numpy as np

from components.expert_load_balancing.elb.data_loader.base_loader import DataType
from components.expert_load_balancing.elb.algorithm_runner.base_algorithm_runner import \
    BaseAlgorithmRunner, AlgorithmType, DEPLOYMENT_JSON_FILE
from components.utils.security_check import check_int
from c2lb import lb_and_intra_layer_affinity_redundancy_deploy
from c2lb_dynamic import lb_redundancy_deploy_for_dynamic
from c2lb_a3 import lb_and_intra_layer_affinity_redundancy_deploy_a3


class StaticC2lbRunner(BaseAlgorithmRunner):
    def __init__(self, args):
        super().__init__(args)
        self.algorithm_type = AlgorithmType.C2LB
    
    def run_algorithm(self, data):
        for period, period_data in data.items():
            if "topk" in period:
                continue
            period_data = process_data(period_data, self.args)
            num_original_expert = period_data.shape[1]
            global_deployment = lb_and_intra_layer_affinity_redundancy_deploy(
                period_data,
                self.args.num_redundancy_expert,
                self.args.num_npus,
                num_original_expert
            )

            global_deployment = format_deployment_to_json(global_deployment)
            output_path = os.path.join(self.args.output_dir, DEPLOYMENT_JSON_FILE.format(period))
            self.save_json(global_deployment, output_path)


class StaticC2lbA3Runner(BaseAlgorithmRunner):
    def __init__(self, args):
        super().__init__(args)
        self.algorithm_type = AlgorithmType.C2LB

    def run_algorithm(self, data):
        if self.args.share_expert_devices != 0:
            raise ValueError("Input incorrect share expert devices parameters.")
        for period, period_data in data.items():
            if "topk" in period:
                continue
            period_data = process_data(period_data, self.args)
            num_original_expert = period_data.shape[1]
            global_deployment = lb_and_intra_layer_affinity_redundancy_deploy_a3(
                period_data,
                self.args.num_redundancy_expert,
                self.args.num_npus,
                num_original_expert
            )

            global_deployment = format_deployment_to_json(global_deployment)
            output_path = os.path.join(self.args.output_dir, DEPLOYMENT_JSON_FILE.format(period))
            self.save_json(global_deployment, output_path)


class DynamicC2lbRunner(BaseAlgorithmRunner):
    def __init__(self, args):
        super().__init__(args)
        self.algorithm_type = AlgorithmType.DYNAMIC_C2LB

    def run_algorithm(self, data):
        for period, period_data in data.items():
            if "topk" in period:
                continue
            period_data = process_data(period_data, self.args)
            global_deployment = lb_redundancy_deploy_for_dynamic(
                period_data,
                self.args.num_redundancy_expert,
                self.args.num_nodes,
                self.args.num_npus
            )

            global_deployment = format_deployment_to_json(global_deployment)
            output_path = os.path.join(self.args.output_dir, DEPLOYMENT_JSON_FILE.format(period))
            self.save_json(global_deployment, output_path)


def process_data(data, args):
    if args.data_type == DataType.MINDIE_SPLITED_CSV or args.data_type == DataType.MINDIE_SPLITED_CSV_WITH_TOPK:
        return process_split_data(data, args)
    return data


def process_split_data(data, args):
    n_layers = args.config_json.get("num_moe_layers", 1)
    check_int(n_layers, min_value=1, param_name="num_moe_layers")
    n_experts = args.config_json.get("num_of_experts", 1)
    check_int(n_experts, min_value=1, param_name="num_of_experts")
    n_share_expert_files = args.config_json.get("num_dangling_shared_experts", args.share_expert_devices)
    check_int(n_share_expert_files, min_value=0, param_name="num_dangling_shared_experts")
    data = data[n_share_expert_files:]
    data = np.concatenate(data, axis=1)
    iteration = data.shape[0] // n_layers
    start_index = n_layers * (iteration - 1)
    end_index = n_layers * iteration
    return data[start_index:end_index]


def format_deployment_to_json(deployment):
    # c2lb算法返回值既有np又有list
    if isinstance(deployment, list):
        deployment = deployment
        try:
            deployment_arr = np.array(deployment)
        except Exception as e:
            raise ValueError("Convert deployment to array failed, please check your input data.") from e
    elif isinstance(deployment, np.ndarray):
        deployment_arr = deployment
        deployment = deployment.tolist()
    else:
        raise ValueError("Illegal deployment type, should be list or np.ndarray.")
    
    if deployment_arr.ndim != 3:
        raise ValueError("Shape of deployment should be 3.")
    
    num_layers, num_ranks, _ = deployment_arr.shape  # dim 2 is experts_num_per_rank
    deployment_json_format = {
        "moe_layer_count": num_layers,
        "layer_list": []
    }

    for layer_idx in range(num_layers):
        deployment_per_layer = {
            "layer_id": layer_idx,
            "device_count": num_ranks,
            "device_list": []
        }
        
        for rank in range(num_ranks):
            deployment_per_rank = {
                "device_id": rank,
                "device_expert": list(deployment[layer_idx][rank])
            }
            deployment_per_layer["device_list"].append(deployment_per_rank)

        deployment_json_format["layer_list"].append(deployment_per_layer)

    return deployment_json_format
