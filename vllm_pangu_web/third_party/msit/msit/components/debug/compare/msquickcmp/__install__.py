# -*- coding: utf-8 -*-
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

import os
import sys
import subprocess
import pkg_resources
from components.utils.install import AitInstaller


class CompareInstall(AitInstaller):
    @staticmethod
    def check():
        check_res = []
        installed_pkg = [pkg.key for pkg in pkg_resources.working_set]

        if "ais-bench" not in installed_pkg:
            check_res.append("[error] msit-benchmark not installed. use `msit install benchmark` to try again")

        if "msit-surgeon" not in installed_pkg:
            check_res.append("[error] msit-surgeon not installed. use `msit install surgeon` to try again")

        if not os.path.exists(os.path.join(os.path.dirname(__file__), "libsaveom.so")):
            check_res.append("[error] build lib saveom.so failed. use `msit build-extra compare` to try again")
        
        if not check_res:
            return "OK"
        else:
            return "\n".join(check_res)

    @staticmethod
    def build_extra(find_links=None):
        if sys.platform == 'win32':
            return
        
        subprocess.run(["/bin/bash", os.path.abspath(os.path.join(os.path.dirname(__file__), "install.sh"))])
