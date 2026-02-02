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

declare -i ret_ok=0
declare -i ret_run_failed=1

WHL_BASE_URL=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('whl_base_url'))")
TOOLS_BAS_URL_SUFFIX=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('tools_base_url'))")
TOOLS_BAS_URL="git+$TOOLS_BAS_URL_SUFFIX"

download_and_install_aclruntime() {
    ACLRUNTIME_VERSION=`pip3 show aclruntime | awk '/Version: /{print $2}'`

    if [ "$arg_force_reinstall" = "--force-reinstall" ]; then
        echo "Force reinstall aclruntime"
    elif [ "$ACLRUNTIME_VERSION" = "0.0.2" ]; then
        echo "aclruntime==0.0.2 already installed, skip"
        return
    fi

    echo "download and install aclruntime"
    PYTHON3_MINI_VERSION=`python3 --version | cut -d'.' -f 2`
    PYTHON3_MINI_VERSION=`python3 --version | cut -d'.' -f 2`
    if [ "$PYTHON3_MINI_VERSION" = "7" ]; then
        SUB_SUFFIX="m"
    else
        SUB_SUFFIX=""
    fi
    echo "PYTHON3_MINI_VERSION=$PYTHON3_MINI_VERSION, SUB_SUFFIX=$SUB_SUFFIX"
    PLATFORM=$(uname -m)

    WHL_NAME="aclruntime-0.0.2-cp3${PYTHON3_MINI_VERSION}-cp3${PYTHON3_MINI_VERSION}${SUB_SUFFIX}-linux_${PLATFORM}.whl"
    echo "WHL_NAME=$WHL_NAME, URL=${WHL_BASE_URL}${WHL_NAME}"
    if [ "$NO_CHECK_CERTIFICATE" == "true" ]; then
        echo "[WARNING] --no-check will skip checking the certificate of the target website, posing security risk."
        wget --no-check-certificate -c "${WHL_BASE_URL}${WHL_NAME}"
    else
        wget -c "${WHL_BASE_URL}${WHL_NAME}"
    fi

    if [ "$PLATFORM" == "aarch64" ]; then
        if [ "$PYTHON3_MINI_VERSION" == "7" ]; then
            sha256Value="2a0e847d382e7afb72819dc9ae287aaaebca0d168a8767a4b10d984f3c451c91"
        elif [ "$PYTHON3_MINI_VERSION" == "8" ]; then
            sha256Value="903f27f1e44d14d2a17cd6f00784562bee90fcfd467beaca014ba8121b6059a8"
        elif [ "$PYTHON3_MINI_VERSION" == "9" ]; then
            sha256Value="fe5e28e3acf08d457953962cc3a8208a500fdc648c710fbc156c3e83bdd7fa4a"
        elif [ "$PYTHON3_MINI_VERSION" == "10" ]; then
            sha256Value="206c2c0e9caa465437465ef2a277bc7f4f3b283cba762cf6ea57d5a7d80aff1e"
        elif [ "$PYTHON3_MINI_VERSION" == "11" ]; then
            sha256Value="a219b54ac9cf46a2cd925f7250339268cab119859beb2cfd4d1fef123eee6378"
        else
            echo "Unsupported python3 version"
            exit 1
        fi
    elif [ "$PLATFORM" == "x86_64" ]; then
        if [ "$PYTHON3_MINI_VERSION" == "7" ]; then
            sha256Value="2cfe9f1f7df767c42d8d2aaff244e930ce2c4dc28eca7be1d2b07376bf4b533d"
        elif [ "$PYTHON3_MINI_VERSION" == "8" ]; then
            sha256Value="0189755441d65b550f0e898c37d3a7eb451ed7c91b560cdb7786acc15b666eb8"
        elif [ "$PYTHON3_MINI_VERSION" == "9" ]; then
            sha256Value="c518c671bf9c2f5a738dd58cb13c7573828d7e98576ed2201dfb2969205f7ddf"
        elif [ "$PYTHON3_MINI_VERSION" == "10" ]; then
            sha256Value="3a8e4162c8d6f0e7fcdfd1e4ae100ae9594eaf0ad1e68e7efd4437cf241a2aab"
        elif [ "$PYTHON3_MINI_VERSION" == "11" ]; then
            sha256Value="0f99823b7a76d621e2cefa59d19e8fc36f3f5f763bb6e9cc0922a96ba34d125e"
        else
            echo "Unsupported python3 version"
            exit 1
        fi
    else
        echo "Unsupported platform"
        exit 1
    fi

    sha256Data=$(sha256sum "$WHL_NAME" | cut -d' ' -f1)
    if [[ "${sha256Data}" != "${sha256Value}" ]]; then
        echo "Failed to verify sha256: $WHL_NAME"
        exit 1
    fi

    pip3 install $WHL_NAME $arg_force_reinstall && rm -f $WHL_NAME
    if [ $? -ne 0 ]; then
        echo "Downloading or installing from whl failed, will install from source code"
        pip3 install -v "${TOOLS_BAS_URL}#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend" --force-reinstall
    fi
}

download_and_install_ais_bench() {
    AIS_BENCH_VERSION=`pip3 show ais_bench | awk '/Version: /{print $2}'`

    if [ "$arg_force_reinstall" = "--force-reinstall" ]; then
        echo "Force reinstall ais_bench"
    elif [ "$AIS_BENCH_VERSION" = "0.0.2" ]; then
        echo "ais_bench==0.0.2 already installed, skip"
        return
    fi

    WHL_NAME="ais_bench-0.0.2-py3-none-any.whl"
    echo "WHL_NAME=$WHL_NAME, URL=${WHL_BASE_URL}${WHL_NAME}"
    if [ "$NO_CHECK_CERTIFICATE" == "true" ]; then
        echo "[WARNING] --no-check will skip checking the certificate of the target website, posing security risk."
        wget --no-check-certificate -c "${WHL_BASE_URL}${WHL_NAME}"
    else
        wget -c "${WHL_BASE_URL}${WHL_NAME}"
    fi

    sha256Value="1d533fc6e48ef0322a163a3294fb640f447a7d461ccb4c840623f9f93082f997"
    sha256Data=$(sha256sum "$WHL_NAME" | cut -d' ' -f1)
    if [[ "${sha256Data}" != "${sha256Value}" ]]; then
        echo "Failed to verify sha256: $WHL_NAME"
        exit 1
    fi

    pip3 install $WHL_NAME $arg_force_reinstall && rm -f $WHL_NAME
    if [ $? -ne 0 ]; then
        echo "Downloading or installing from whl failed, will install from source code"
        pip3 install -v "${TOOLS_BAS_URL}#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend" --force-reinstall
    fi
}

ret=0
download_and_install_aclruntime
ret=$(( $ret + $? ))

download_and_install_ais_bench
ret=$(( $ret + $? ))

exit $ret