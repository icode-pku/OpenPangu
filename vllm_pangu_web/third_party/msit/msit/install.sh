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
arg_force_reinstall=
only_compare=
only_surgen=
only_benchmark=
only_analyze=
only_convert=
only_profile=
only_llm=
only_tensor_view=
only_graph=
arg_help=0

while [[ "$#" -gt 0 ]]; do case $1 in
  --force-reinstall) arg_force_reinstall=--force-reinstall;;
  -f) arg_force_reinstall=--force-reinstall;;
  --full) full_install=--full;;
  --compare) only_compare=true;;
  --surgeon) only_surgeon=true;;
  --benchmark) only_benchmark=true;;
  --analyze) only_analyze=true;;
  --convert) only_convert=true;;
  --profile) only_profile=true;;
  --llm) only_llm=true;;
  --tensor-view) only_tensor_view=true;;
  --graph) only_graph=true;;
  --uninstall) uninstall=true;;
  -y) all_uninstall=-y;;
  -h|--help) arg_help=1;;
  *) echo "Unknown parameter: $1";exit 1;
esac; shift; done

if [ ! "$(command -v python3)" ]
then
  echo "Error: python3 is not installed" >&2
  exit 1;
fi

if [ ! "$(command -v pip3)" ]; then
  echo "Error: pip3 is not installed" >&2
  exit 1;
fi

if [ "$arg_help" -eq "1" ]; then
  echo "Usage: $0 [options]"
  echo " --help or -h : Print help menu"
  echo " --surgeon : only install debug surgeon component"
  echo " --compare : only install debug compare component"
  echo " --benchmark : only install benchmark component"
  echo " --analyze : only install analyze component"
  echo " --convert : only install convert component"
  echo " --profile : only install profile component"
  echo " --llm : only install llm component"
  echo "--tensor-view: only install tensor-view component"
  echo "--graph: only install graph component"
  echo " --full : using with install, install all components and dependencies, may need sudo privileges"
  echo " --uninstall : uninstall"
  echo " -y : using with uninstall, don't ask for confirmation of uninstall deletions"
  exit;
fi

# 若pip源为华为云，则优先安装skl2onnx(当前mirrors.huaweicloud.com中skl2onnx已停止更新，不包含1.14.1及以上版本)
pre_check_skl2onnx(){
  http_mirror_url=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('http_mirror_url'))")
  https_mirror_url=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('https_mirror_url'))")
  pip_source_index_url=$(pip3 config list | grep index-url | awk -F'=' '{print $2}' | tr -d "'")
  if [ "${pip_source_index_url}" == "${http_mirror_url}" ] || [ "${pip_source_index_url}" == "${https_mirror_url}" ]
  then
    mirror_tools_url=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('mirror_tools_url'))")
    pip3 install skl2onnx==1.14.1 -i $mirror_tools_url --trusted-host mirrors.tools.huawei.com
  fi
}


uninstall(){
  pip3 uninstall msit analyze_tool convert_tool compare auto_optimizer msprof ${all_uninstall}
  if [ -z $only_debug ] && [ -z $only_compare ] && [ -z $only_surgen ] && [ -z $only_benchmark ] && [ -z $only_analyze ] && [ -z $only_convert ] && [ -z $only_profile ] && [ -z $only_llm ] && [ -z $only_tensor_view ]
  then
    pip3 uninstall msit msit-analyze aclruntime ais_bench msit-benchmark msit-convert msit-compare msit-surgeon msit-profile msit-llm msit-tensor-view msit-graph ${all_uninstall}
  else
    if [ ! -z $only_compare ]
    then
      pip3 uninstall msit-compare ${all_uninstall}
    fi

    if [ ! -z $only_surgeon ]
    then
      pip3 uninstall msit-surgeon ${all_uninstall}
    fi

    if [ ! -z $only_benchmark ]
    then
      pip3 uninstall aclruntime ais_bench ${all_uninstall}
      pip3 uninstall msit-benchmark ${all_uninstall}
    fi

    if [ ! -z $only_analyze ]
    then
      pip3 uninstall msit-analyze ${all_uninstall}
    fi

    if [ ! -z $only_convert ]
    then
      pip3 uninstall msit-convert ${all_uninstall}
    fi

    if [ ! -z $only_profile ]
    then
      pip3 uninstall msit-profile ${all_uninstall}
    fi

    if [ ! -z $only_llm ]
    then
      pip3 uninstall msit-llm ${all_uninstall}
    fi

    if [ ! -z $only_tensor_view ]
    then
      pip3 uninstall msit-tensor-view ${all_uninstall}
    fi

    if [ ! -z $only_graph ]
    then
      pip3 uninstall msit-graph ${all_uninstall}
    fi
  fi
  exit;
}


build_opchecker_so() {
    echo ""
    echo "Try building libatb_speed_torch.so for msit llm."
    cd ${CURRENT_DIR}/components/llm/msit_llm/opcheck/atb_operators
    bash build.sh
    cd -
    echo ""
}


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
  g++ ${CURRENT_DIR}/components/debug/compare/msquickcmp/save_om_model/export_om_model.cpp \
          -I ${ge_dev_path}/include \
          -L ${ge_dev_path}/lib64 \
          -lge_compiler $COMPILE_OPTIONS \
          --std=c++11 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o libsaveom.so
  
  if [ ! -f "${CURRENT_DIR}/libsaveom.so" ]
  then
    echo "libsaveom.so compilation failed"
  else
    if [ ! -d "${SITE_PACKAGES_PATH}/msquickcmp/" ]
    then
      rm libsaveom.so
      echo "msquickcmp not exist, failed to install libsaveom.so"
    else
      mv libsaveom.so "${SITE_PACKAGES_PATH}/msquickcmp/"
      echo "Finish libsaveom.so installation."
    fi
  fi
}


install(){
  pip3 install ${CURRENT_DIR} ${arg_force_reinstall}

  if [ ! -z $only_compare ]
  then
    only_benchmark=true;
    only_surgeon=true;
    pre_check_skl2onnx
    pip3 install ${CURRENT_DIR}/components/debug/compare ${arg_force_reinstall}

    build_om_so

  fi

  if [ ! -z $only_surgeon	 ]
  then
    pip3 install ${CURRENT_DIR}/components/debug/surgeon ${arg_force_reinstall}

  fi

  if [ ! -z $only_benchmark ]
  then
    bash ${CURRENT_DIR}/components/benchmark/msit_benchmark/install.sh
    pip3 install ${CURRENT_DIR}/components/benchmark ${arg_force_reinstall}
  fi

  if [ ! -z $only_analyze ]
  then
    pip3 install ${CURRENT_DIR}/components/analyze ${arg_force_reinstall}
  fi

  if [ ! -z $only_convert ]
  then
    pip3 install ${CURRENT_DIR}/components/convert ${arg_force_reinstall}

    bash ${CURRENT_DIR}/components/convert/build.sh
  fi

  if [ ! -z $only_profile ]
  then
    pip3 install ${CURRENT_DIR}/components/profile ${arg_force_reinstall}
  fi

  if [ ! -z $only_llm ]
  then
      pip3 install ${CURRENT_DIR}/components/llm ${arg_force_reinstall}
      build_opchecker_so
  fi

  if [ ! -z $only_tensor_view ]
    then
        pip3 install ${CURRENT_DIR}/components/tensor_view ${arg_force_reinstall}
        build_opchecker_so
    fi

  if [ -z $only_compare ] && [ -z $only_surgeon ] && [ -z $only_benchmark ] && [ -z $only_analyze ] && [ -z $only_convert ] && [ -z $only_profile ] && [ -z $only_llm ] && [ -z $only_tensor_view ]
  then
    pre_check_skl2onnx

    pip3 install ${CURRENT_DIR}/components/debug/compare \
    ${CURRENT_DIR}/components/debug/surgeon \
    ${CURRENT_DIR}/components/benchmark \
    ${CURRENT_DIR}/components/analyze \
    ${CURRENT_DIR}/components/convert \
    ${CURRENT_DIR}/components/profile \
    ${CURRENT_DIR}/components/llm \
    ${CURRENT_DIR}/components/tensor_view \
    ${arg_force_reinstall}

    bash ${CURRENT_DIR}/components/benchmark/msit_benchmark/install.sh
    bash ${CURRENT_DIR}/components/convert/build.sh

    build_om_so
    build_opchecker_so
  fi

  rm -rf ${CURRENT_DIR}/msit.egg-info
}


if [ ! -z $uninstall ]
then
  uninstall
else
  install
fi