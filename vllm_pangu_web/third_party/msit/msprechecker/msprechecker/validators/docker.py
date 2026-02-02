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

import shutil
import subprocess

from .base import BaseValidator


class DockerValidator(BaseValidator):
    @staticmethod
    def validate(actual_value, expected_value) -> bool:
        if expected_value == "image_exists":
            if not shutil.which('/usr/bin/docker'):
                return False
            
            cmds = ['docker', 'image', 'inspect', f"{actual_value}"]
            try:
                subprocess.check_call(
                    cmds,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3
                )
            except Exception:
                return False

            return True
        return False
