# Copyright (c) 2023-2025 Huawei Technologies Co., Ltd.
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

import sys
from collections import namedtuple
from unittest.mock import patch, MagicMock
import pytest

ACL_RUNTIME = namedtuple("aclruntime", "key")("aclruntime")
AIS_BENCH_RUNTIME = namedtuple("ais_bench", "key")("ais-bench")


def test_install_check_given_all_installed_then_pass():
    mock_pkg = MagicMock()
    mock_pkg.working_set = [ACL_RUNTIME, AIS_BENCH_RUNTIME]
    with patch.dict(sys.modules, {"pkg_resources": mock_pkg}):
        from msit_benchmark.__install__ import BenchmarkInstall
        assert BenchmarkInstall().check() == "OK"


@pytest.mark.filterwarnings("ignore:Deprecated call to:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:pkg_resources is deprecated:DeprecationWarning")
@patch("subprocess.run")
def test_install_build_extra_given_valid_then_pass(mock_run):
    from msit_benchmark.__install__ import BenchmarkInstall
    BenchmarkInstall().build_extra()
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_install_download_extra_given_valid_then_pass(mock_run):
    from msit_benchmark.__install__ import BenchmarkInstall
    BenchmarkInstall().download_extra(dest=".")
    mock_run.assert_called_once()
