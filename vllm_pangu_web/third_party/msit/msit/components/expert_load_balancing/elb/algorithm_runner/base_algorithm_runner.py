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
from enum import Enum
import json

from components.utils.file_open_check import ms_open
from components.utils.security_check import get_valid_write_path


class AlgorithmType(Enum):
    C2LB = 0
    SPECULATIVE_MOE_LEVEL_1 = 1
    DYNAMIC_C2LB = 2
    SPECULATIVE_MOE_LEVEL_2 = 3
    SPECULATIVE_MOE_LEVEL_1_MIXED = 4
    SPECULATIVE_MOE_LEVEL_2_MIXED = 5
    UNKNOWN_TYPE = 999


DEPLOYMENT_JSON_FILE = "{}_global_deployment.json"


class BaseAlgorithmRunner:
    def __init__(self, args):
        self.algorithm_type = AlgorithmType.UNKNOWN_TYPE
        self.args = args
    
    @staticmethod
    def save_json(data, output_path, default=None):
        output_path = get_valid_write_path(output_path)
        with ms_open(output_path, "w") as json_file:
            json.dump(data, json_file, indent=4, default=default)

    def run_algorithm(self, data):
        pass
            