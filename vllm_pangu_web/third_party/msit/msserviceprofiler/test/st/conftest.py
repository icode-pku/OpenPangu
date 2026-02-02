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
import logging
import os
import shutil
import uuid

import pytest
from pytest_check import check_functions, check

pytest.assume = check_functions.is_true


def pytest_addoption(parser):
    parser.addoption("--device", action="append", type=int, default=[], help="devices")
    parser.addoption(
    "--mindie-path", action="store", default="/usr/local/Ascend/mindie/latest/mindie-service", help="mindie path"
    )
    parser.addoption("--dataset-path", action="store", default="/dataset", help="dataset path")
    parser.addoption("--model-path", action="store", default="/model", help="model path")
    parser.addoption("--workspace", action="store", default="/workspace", help="workspace path")
    parser.addoption("--debug-mode", action="store", type=bool, default=False, help="debug")


@pytest.fixture(scope="session")
def devices(request):
    devices = request.config.getoption("--device")
    return devices if devices else [0]


@pytest.fixture(scope="session")
def mindie_path(request):
    return request.config.getoption("--mindie-path")


@pytest.fixture(scope="session")
def dataset_path(request):
    return request.config.getoption("--dataset-path")


@pytest.fixture(scope="session")
def model_path(request):
    return request.config.getoption("--model-path")


@pytest.fixture(scope="session")
def workspace(request):
    workspace_path = request.config.getoption("--workspace")
    os.makedirs(workspace_path, exist_ok=True)
    return workspace_path


@pytest.fixture(scope="session")
def smokedata(workspace):
    return os.path.join(workspace, "smokedata")


@pytest.fixture(scope="session")
def debug(request):
    return request.config.getoption("--debug-mode")


@pytest.fixture(scope="function")
def tmp_workspace(debug):
    index = str(uuid.uuid4())
    workspace_path = os.path.join("/tmp/server-smoke", index)
    os.makedirs(workspace_path, exist_ok=True)
    yield workspace_path
    if not debug:
        shutil.rmtree(workspace_path, ignore_errors=True)
    else:
        logging.info(f"tmp workspace is {workspace_path}")


@pytest.fixture(scope="session")
def smoke_args(devices, mindie_path, workspace, smokedata, debug):
    return dict(devices=devices, mindie_path=mindie_path, smokedata=smokedata, workspace=workspace, debug=debug)