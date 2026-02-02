#!/bin/bash
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

if [ ! "$(command -v pip3)" ];then
  echo "Error: pip3 is not installed."
  exit 1
fi

PIP3=$(readlink -f $(which pip3))
PIP3_DIR=$(dirname ${PIP3})
PYTHON=${PIP3_DIR}/python
if [ ! -f ${PYTHON} ];then
  PIP_HEAD=$(head -n 1 ${PIP3})
  PYTHON=${PIP_HEAD:2}
fi

MODEL_CONVERT_PATH=$(dirname $(${PYTHON} -c "import model_convert;print(model_convert.__file__)"))
CUR_PATH=$(dirname $(readlink -f $0))
