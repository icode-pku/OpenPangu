#!/bin/bash
# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
CURRENT_DIR=$(dirname $(readlink -f $0))


build_opchecker_so() {
    echo ""
    echo "Try building libatb_speed_torch.so for msit llm. If not using opcheck, ignore errors if any"
    cd ${CURRENT_DIR}/opcheck/atb_operators
    bash build.sh
    cd -
    echo ""
}

build_opchecker_so
