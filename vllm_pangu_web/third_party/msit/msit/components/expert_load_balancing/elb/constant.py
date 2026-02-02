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

PREFILL = "prefill"
DECODE = "decode"
DECODE_FILE_NAME = "decode_global_deployment"
PREFILL_FILE_NAME = "prefill_global_deployment" 

ALGORITHM_C2LB = "0"
ALGORITHM_SPECULATIVE_MOE_LEVEL_1 = "1"
ALGORITHM_DYNAMIC_C2LB = "2"
ALGORITHM_SPECULATIVE_MOE_LEVEL_2 = "3"
ALGORITHM_SPECULATIVE_MOE_LEVEL_1_MIXED = "4"
ALGORITHM_SPECULATIVE_MOE_LEVEL_2_MIXED = "5"

A2 = "a2"
A3 = "a3"

SUPPORTED_COMBINATIONS = {
    "a2": {"1", "3", "4", "5"},
    "a3": {"1", "3"},
}

SPECULATIVE_MOE_ALGORITHM = {
        ALGORITHM_SPECULATIVE_MOE_LEVEL_1,
        ALGORITHM_SPECULATIVE_MOE_LEVEL_2,
        ALGORITHM_SPECULATIVE_MOE_LEVEL_1_MIXED,
        ALGORITHM_SPECULATIVE_MOE_LEVEL_2_MIXED
    }