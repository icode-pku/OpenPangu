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

CURRENT_DIR=$(dirname $(readlink -f $0))

build_om_so() {
  echo "Installing libsaveom.so"
  echo "This part is used for the accuracy comparison of mindir and onnx models. "
  echo "If installation failed, the usage of other components will not be affected."
  SITE_PACKAGES_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")

  if [ "$ASCEND_TOOLKIT_HOME" != "" ]; then
      toolkit_home=$ASCEND_TOOLKIT_HOME
  else
      toolkit_home=$ASCEND_AICPU_PATH
  fi

  ge_dev_path=$toolkit_home/$(uname -m)-linux/
  if [ ! -e "$ge_dev_path/include" ] || [ ! -e "$ge_dev_path/lib64" ]; then
      echo "[WARNING] include or lib64 not found in ge_dev_path=$ge_dev_path, try installing CANN toolkit if comparing mindir and onnx models"
      return
  fi

  COMPILE_OPTIONS="-Wl,-z,relro,-z,now,-z,noexecstack -s -fstack-protector-all -ftrapv -D_FORTIFY_SOURCE=2"
  g++ ${CURRENT_DIR}/save_om_model/export_om_model.cpp \
          -I ${ge_dev_path}/include \
          -L ${ge_dev_path}/lib64 \
          -lge_compiler $COMPILE_OPTIONS \
          --std=c++11 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o ${CURRENT_DIR}/libsaveom.so
  
  if [ ! -f "${CURRENT_DIR}/libsaveom.so" ]
  then
    echo "libsaveom.so compilation failed"
  else
    echo "libsaveom.so compilation successed"
    if [ ! -d "${SITE_PACKAGES_PATH}/msquickcmp/" ]
    then
      rm libsaveom.so
      echo "msquickcmp not exist, failed to install libsaveom.so"
    else
      cp libsaveom.so "${SITE_PACKAGES_PATH}/msquickcmp/"
      echo "Finish libsaveom.so installation."
    fi
  fi
}


build_om_so
