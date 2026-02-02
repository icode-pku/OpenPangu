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


CUR_DIR=$(dirname $(readlink -f $0))
COMPONENTS_DIR=${CUR_DIR}/../../components
TOP_DIR=${COMPONENTS_DIR}/..
ALL_VALID_TEST_MODULE=(benchmark convert graph profile tensor_view utils llm debug surgeon analyze)


install_packages() {
    PYTHON3_VERSION=$(python3 --version | cut -d '.' -f 2)
    if ! pip show pytest &> /dev/null; then
        echo "pytest not found, trying to install..."
        pip install pytest
    fi

    if ! pip show pytest-cov &> /dev/null; then
        echo "pytest-cov not found, trying to install..."
        pip install pytest-cov
    fi

    if [[ "$PYTHON3_VERSION" = "7" ]]; then
        pip install numpy==1.21.6
    elif [[ "$PYTHON3_VERSION" = "8" ]]; then
        pip install numpy==1.24.4
    else
        pip install numpy==1.26.4
    fi
}

init_msit_env() {
    export PYTHONPATH=${TOP_DIR}:${PYTHONPATH}
    export PYTHONPATH=${COMPONENTS_DIR}/analyze:${PYTHONPATH}
    export PYTHONPATH=${COMPONENTS_DIR}/benchmark:${PYTHONPATH}
    export PYTHONPATH=${COMPONENTS_DIR}/convert:${PYTHONPATH}
    export PYTHONPATH=${COMPONENTS_DIR}/debug/compare:${PYTHONPATH}
    export PYTHONPATH=${COMPONENTS_DIR}/debug/opcheck:${PYTHONPATH}
    export PYTHONPATH=${COMPONENTS_DIR}/debug/surgeon:${PYTHONPATH}
    export PYTHONPATH=${COMPONENTS_DIR}/expert_load_balancing:${PYTHONPATH}
    export PYTHONPATH=${COMPONENTS_DIR}/graph:${PYTHONPATH}
    export PYTHONPATH=${COMPONENTS_DIR}/llm:${PYTHONPATH}
    export PYTHONPATH=${COMPONENTS_DIR}/profile:${PYTHONPATH}
    export PYTHONPATH=${COMPONENTS_DIR}/tensor_view:${PYTHONPATH}
}

run_single_module_ut() {
    ut_dir=${1}
    source_code_dir=${2}
    python3 -m pytest ${ut_dir} --cov-config ${CUR_DIR}/.coveragerc \
    --cov=${source_code_dir} --cov-branch \
    --cov-report=html:${CUR_DIR}/report/htmlcov --cov-report=xml:${CUR_DIR}/report/xmlcov
}

run_all() {
    python3 -m pytest ${CUR_DIR} --cov-config ${CUR_DIR}/.coveragerc \
    --cov=${COMPONENTS_DIR} --cov-branch \
    --cov-report=html:${CUR_DIR}/report/htmlcov --cov-report=xml:${CUR_DIR}/report/xmlcov
}

main() {
    install_packages
    init_msit_env
    dt_mode=${1:-"normal"}
    if [[ $dt_mode == "normal" ]]; then
        run_all
    else
        for case in ${ALL_VALID_TEST_MODULE[@]}; do
            if [[ "$dt_mode" =~ "$case" ]]; then
                echo "run test module: "${case}
                ut_dir=${CUR_DIR}/${case}_ut
                cov_dir=${COMPONENTS_DIR}/${case}
                run_single_module_ut $ut_dir $cov_dir
            fi
        done
    fi
}

main "$@"