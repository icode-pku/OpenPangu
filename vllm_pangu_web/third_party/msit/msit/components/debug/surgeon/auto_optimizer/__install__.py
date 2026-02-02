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
from components.utils.install import AitInstaller
import pkg_resources


class SurgeonInstall(AitInstaller):
    @staticmethod
    def check():
        check_res = []
        installed_pkg = [pkg.key for pkg in pkg_resources.working_set]

        if "ais-bench" not in installed_pkg:
            check_res.append("[warnning] msit-benchmark not installed. will make the inference feature unusable. "
                             "use `msit install benchmark` to try again")

        if not check_res:
            return "OK"
        else:
            return "\n".join(check_res)
