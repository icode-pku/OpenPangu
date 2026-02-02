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
set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd)
AIT_LLM_INSTALL_PATH="$(python3 -c 'import msit_llm, os; print(os.path.dirname(os.path.abspath(msit_llm.__file__)))')"
IGNORE_INFO="If not using opcheck, ignore this error."

echo SCRIPT_DIR: $SCRIPT_DIR

function check_nlohmann_json()
{
    jsonVERSION=$1
    filePath=$2

    if [[ "$jsonVERSION" == "3.11.1" ]]; then
        sha256Value="598becb62ee0e01cf32795073c8ae09b6e95335cd43a4417b785d93ce105b0d0"
    elif [[ "$jsonVERSION" == "3.11.2" ]]; then
        sha256Value="d69f9deb6a75e2580465c6c4c5111b89c4dc2fa94e3a85fcd2ffcd9a143d9273"
    elif [[ "$jsonVERSION" == "3.11.3" ]]; then
        sha256Value="0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406"
    else
        echo "Unknown json version"
        exit 1
    fi

    sha256Data=$(sha256sum "${filePath}" | cut -d' ' -f1)
    if [[ "${sha256Data}" != "${sha256Value}" ]]; then
        echo "Failed to verify sha256: ${filePath}"
        exit 1
    fi
}

function download_nlohmann_json()
{
    JSON_BASE_URL=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('json_base_url'))")
    if [[ "$NLOHMAN_JSON_LINE" =~ "v3_11_1" ]]; then
        JSON_VERSION="3.11.1"
    elif [[ "$NLOHMAN_JSON_LINE" =~ "v3_11_2" ]]; then
        JSON_VERSION="3.11.2"
    elif [[ "$NLOHMAN_JSON_LINE" =~ "v3_11_3" ]]; then
        JSON_VERSION="3.11.3"
    else
        echo "[WARNING] Cannot find nlohmann json version, will treat as 3.11.2"
        JSON_VERSION="3.11.2"
    fi
    JSON_TAR="v${JSON_VERSION}.tar.gz"
    JSON_URL="$JSON_BASE_URL/$JSON_TAR"
    echo "JSON_URL=$JSON_URL"

    if [[ -d "$SCRIPT_DIR/dependency/nlohmann" && -e "$SCRIPT_DIR/dependency/$JSON_TAR" ]]; then
        check_nlohmann_json ${JSON_VERSION} $SCRIPT_DIR/dependency/$JSON_TAR
        echo "[INFO] $SCRIPT_DIR/dependency/nlohmann already exists, skip downloading"
        return
    fi

    mkdir -p $SCRIPT_DIR/dependency
    cd $SCRIPT_DIR/dependency
    rm -f *.tar.gz
    if [ "$AIT_INSTALL_FIND_LINKS" != "" ]; then 
        cp "$AIT_INSTALL_FIND_LINKS/$JSON_TAR" ./
    else
        wget -c $JSON_URL
    fi 
    check_nlohmann_json ${JSON_VERSION} $JSON_TAR

    if [ "$AIT_DOWNLOAD_PATH" != "" ]; then 
        mv $JSON_TAR "$AIT_DOWNLOAD_PATH"
        cd -
        exit 0
    fi
    tar xf $JSON_TAR

    JSON_FILE_NAME="json-$JSON_VERSION"
    if [ ! -d $JSON_FILE_NAME ] ; then
        echo "[ERROR] $JSON_FILE_NAME not exists. Check if anything wrong with downloading. $IGNORE_INFO"
        exit 1
    fi
    rm -rf ./nlohmann
    mv $JSON_FILE_NAME/include/nlohmann ./

    rm -rf $JSON_FILE_NAME
    cd -
}

if [ "$AIT_DOWNLOAD_PATH" != "" ]; then 
    download_nlohmann_json
    exit
fi

if [ "$AIT_LLM_INSTALL_PATH" == "" ]; then
    echo "[ERROR] msit_llm not found in python packages. Make sure msit_llm is installed for pip. $IGNORE_INFO"
    exit 1
fi

if [ "$ASCEND_TOOLKIT_HOME" == "" ]; then
    echo "[ERROR] ASCEND_TOOLKIT_HOME is empty. Make sure CANN toolkit is installed correctly. $IGNORE_INFO"
    exit 1
fi

if [ "$ATB_HOME_PATH" == "" ]; then
    echo "[ERROR] ATB_HOME_PATH is empty. Make sure atb is installed correctly. $IGNORE_INFO"
    exit 1
fi

if [ "$ATB_SPEED_HOME_PATH" == "" ]; then
    echo "[ERROR] ATB_SPEED_HOME_PATH is empty. Make sure mindie_atb_models is configured correctly. $IGNORE_INFO"
    exit 1
fi

if [ ! -e "$ATB_HOME_PATH/lib/libatb.so" ]; then
    echo "[ERROR] $ATB_HOME_PATH/lib/libatb.so not exists. Make sure atb is installed correctly. $IGNORE_INFO"
    exit 1
fi

NLOHMAN_JSON_LINE=`nm -D $ATB_HOME_PATH/lib/libatb.so | grep -i nlohmann | head -n 1`
if [ "$NLOHMAN_JSON_LINE" = "" ]; then
    echo "nlohmann json info not found in $ATB_HOME_PATH/lib/libatb.so. This shouldn't happen. make sure atb is installed correctly"
fi

if [[ "$NLOHMAN_JSON_LINE" =~ "cxx11" ]]; then
    CMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
else
    CMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
fi
echo "CMAKE_CXX_FLAGS=$CMAKE_CXX_FLAGS"


download_nlohmann_json

ATB_VERSION=`python3 -c '
import os
from msit_llm.common.utils import check_data_file_size
from components.utils.file_open_check import ms_open
from components.utils.constants import TENSOR_MAX_SIZE
version = "8.0.RC3.B020" # Default value
atb_version_file = os.path.abspath(os.path.join(os.getenv("ATB_HOME_PATH", ""), "..", "..", "version.info"))
if os.path.exists(atb_version_file) and os.path.isfile(atb_version_file) and check_data_file_size(atb_version_file):
    with ms_open(atb_version_file, max_size=TENSOR_MAX_SIZE) as ff:
        for ii in ff.readlines():
            if "version" in ii.lower():
                version = ii.split(":")[-1]
                break

# "8.0.RC3.B020" -> [8, 0, 3, 020] -> 8 * 1e9 + 0 * 1e6 + 3 * 1e3 + 020 -> 8000003020
index_exps = [1e9, 1e6, 1e3, 1e0]
version_num = 0
for tok, index_exp in zip(version.split("."), index_exps):
    digit_tok = int("".join([ii for ii in tok if str.isdigit(ii)]) or 0)
    version_num += int(digit_tok * index_exp)
print(version_num)
'`
echo "ATB_VERSION=$ATB_VERSION"

if [ -d "$SCRIPT_DIR/build" ]; then
    rm -rf $SCRIPT_DIR/build
fi

mkdir -p $SCRIPT_DIR/build

cd $SCRIPT_DIR/build
cmake .. -DCMAKE_INSTALL_PREFIX=$AIT_LLM_INSTALL_PATH/opcheck -DCMAKE_CXX_FLAGS=$CMAKE_CXX_FLAGS -DATB_VERSION=$ATB_VERSION
make -j4 && make install
echo "[INFO] Copied $PWD/build/libopchecker.so -> $AIT_LLM_INSTALL_PATH/opcheck/libopchecker.so"
cd -
