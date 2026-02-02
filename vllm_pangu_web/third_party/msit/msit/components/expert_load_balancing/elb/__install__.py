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

import os
import sys
import subprocess
import pkg_resources
from components.utils.install import AitInstaller


class ExpertLoadBalanceInstall(AitInstaller):
    @staticmethod
    def check():
        check_res = []
        installed_pkg = [pkg.key for pkg in pkg_resources.working_set]

        if "ortools" not in installed_pkg:
            check_res.append("[error] ortools not installed. Please install ortools packages.")

        if not check_res:
            return "OK"
        else:
            return "\n".join(check_res)