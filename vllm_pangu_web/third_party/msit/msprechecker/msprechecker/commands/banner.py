# -*- coding: utf-8 -*-
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
import shutil
import platform
from abc import ABC, abstractmethod

from ..collectors import AscendCollector, LscpuCollector
from ..utils import get_pkg_version, get_npu_type, get_npu_count, global_logger


class InfoSection(ABC):
    @abstractmethod
    def get_info(self):
        pass


class PlatformInfoSection(InfoSection):
    def get_info(self):
        return f"Platform: {platform.platform()}"


class PythonInfoSection(InfoSection):
    def __init__(self, packages):
        self.packages = packages

    def get_info(self):
        python_info = f"Python {platform.python_version()}"
        for package in self.packages:
            package_ver = get_pkg_version(package)
            if package_ver:
                python_info += f", {package}-{package_ver}"
            else:
                python_info += f", {package} [not installed]"
        return python_info


class CpuInfoSection(InfoSection):
    def get_info(self):
        ret = LscpuCollector().collect()
        data = ret.data
        model_name = data.get('model_name', 'Unknown Type')
        return f"CPU: {model_name} ({os.cpu_count()} cores)"


class NpuInfoSection(InfoSection):
    def get_info(self):
        npu_type, npu_device_nums = get_npu_type()
        npu_type = npu_type.display if npu_type else "Unknown Type"
        npu_device_nums = npu_device_nums or 0
        npu_count = get_npu_count()
        return f"NPU: {npu_type} ({npu_device_nums} devices {npu_count} chips)"


class AscendInfoSection(InfoSection):
    def get_info(self):
        ret = AscendCollector().collect()
        data = ret.data
        if not data:
            return "Ascend: not found"
        return self._format_ascend_info(data)

    def _format_ascend_info(self, data):
        prefix = "Ascend: "
        indent = " " * len(prefix)
        items = list(data.items())
        lines = []
        for idx, (comp, info) in enumerate(items):
            line_prefix = prefix if idx == 0 else indent
            lines.append(f"{line_prefix}{comp}: {self._format_version(info)}")
        return "\n".join(lines)

    def _format_version(self, info):
        comp_ver = info.get('version', "not found")
        timestamp = info.get('timestamp')
        if timestamp:
            comp_ver += f" ({timestamp})"
        commit_id = info.get('commit')
        if commit_id:
            comp_ver += f" -- {commit_id}"
        return comp_ver


class BannerPresenter:
    PYTHON_INFO_PACKAGES = ['msprechecker', 'torch', 'torch_npu', 'transformers']

    def __init__(self, *, sections=None):
        self.sections = sections or [
            PlatformInfoSection(),
            PythonInfoSection(self.PYTHON_INFO_PACKAGES),
            CpuInfoSection(),
            NpuInfoSection(),
            AscendInfoSection()
        ]

    def add_section(self, section: InfoSection):
        self.sections.append(section)
    
    def print_banner(self):
        cols, _ = shutil.get_terminal_size()

        title = "MindStudio Prechecker Tool"
        global_logger.info(f" {title} ".center(cols, "="))

        for section in self.sections:
            global_logger.info(section.get_info())

        global_logger.info("-" * cols)
