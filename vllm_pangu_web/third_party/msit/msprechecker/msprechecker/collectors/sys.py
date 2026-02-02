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
import re
import shlex
import platform
import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import psutil
from msguard.security import open_s

from .base import BaseCollector


# *** LSCPU ***
class LscpuCollector(BaseCollector):
    MODEL_NAME = "model_name"
    
    # 将翻译映射和优先级逻辑整合在一起
    KEY_MAPPINGS = {
        # 主键优先
        "Model name": MODEL_NAME,
        "型号名称": MODEL_NAME,
        # 备选键（仅在主键未找到时使用）
        "BIOS Model name": MODEL_NAME
    }
    
    # 主键集合，用于判断是否已经找到主要信息
    PRIMARY_KEYS = {"Model name", "型号名称"}

    def _parse_output(self, output):
        info = {}
        found_primary_info = {self.MODEL_NAME: False}

        for line in output.splitlines():
            if ':' not in line:
                continue
                
            key, value = [x.strip() for x in line.split(':', 1)]
            
            if key in self.KEY_MAPPINGS:
                target_field = self.KEY_MAPPINGS[key]
                
                # 如果是主键或者对应字段还未设置
                if (key in self.PRIMARY_KEYS or 
                    not found_primary_info.get(target_field, False)):
                    info[target_field] = value
                    
                    # 标记主键是否已找到
                    if key in self.PRIMARY_KEYS:
                        found_primary_info[target_field] = True
        
        return info

    def _collect_data(self):
        try:
            output = subprocess.check_output(
                ['/usr/bin/lscpu'], 
                stderr=subprocess.DEVNULL, 
                text=True
            )
        except Exception as e:
            self.error_handler.add_error(
                filename=__file__,
                function='_collect_data',
                lineno=72,
                what="执行命令失败：'/usr/bin/lscpu'",
                reason=str(e)
            )
            return {}
        
        return self._parse_output(output)


# *** Virtual Machine ***
class VirtualMachineCollector(BaseCollector):
    CPU_INFO_PATH = "/proc/cpuinfo"
    HYPERVISOR_KEYWORDS = ["hypervisor", "vmware", "virtualbox", "kvm", "xen"]

    def _collect_data(self):
        try:
            return self._parse_content()
        except Exception as e:
            self.error_handler.add_error(
                filename=__file__,
                function='_collect_data',
                lineno=97,
                what=f"打开文件失败：{self.CPU_INFO_PATH}",
                reason=str(e)
            )
            return {}
    
    def _parse_content(self):
        with open_s(self.CPU_INFO_PATH) as f:
            if any(self._check_hypervisor_keyword(line) for line in f):
                return {"virtual_machine": True}

        return {"virtual_machine": False}
    
    def _check_hypervisor_keyword(self, line):
        return any(keyword in line.lower() for keyword in self.HYPERVISOR_KEYWORDS)


# *** CPU High Performance ***
class CPUHighPerformanceStrategy(ABC):
    """Abstract base class for CPU high performance detection strategies."""
    @abstractmethod
    def check(self):
        """"""


class PsutilStrategy(CPUHighPerformanceStrategy):
    def check(self):
        cpu_freq = psutil.cpu_freq()
        if not cpu_freq:
            return False
        return cpu_freq.current == cpu_freq.max


class DmidecodeStrategy(CPUHighPerformanceStrategy):
    DMIDECODE_CMD = shlex.split("dmidecode -t processor")
    DMIDECODE_PATTERNS = (
        re.compile(r'Max Speed:\s*([^\n]+)', re.IGNORECASE),
        re.compile(r'Current Speed:\s*([^\n]+)', re.IGNORECASE)
    )

    def check(self):
        max_speeds = []
        current_speeds = []
        try:
            dmi_output = subprocess.check_output(
                self.DMIDECODE_CMD,
                stderr=subprocess.DEVNULL,
                text=True
            )
        except Exception:
            return False

        max_pattern, cur_pattern = self.DMIDECODE_PATTERNS
        for line in dmi_output.splitlines():
            m_max = max_pattern.search(line)
            m_cur = cur_pattern.search(line)
            if m_max:
                max_speeds.append(m_max.group(1).strip())
            if m_cur:
                current_speeds.append(m_cur.group(1).strip())
        return bool(max_speeds and current_speeds and max_speeds == current_speeds)


class CpupowerStrategy(CPUHighPerformanceStrategy):
    CPUPOWER_CMD = shlex.split("cpupower frequency-info")
    CPUPOWER_PATTERNS = (
        re.compile(r'hardware limits:\s*[\d\.]+\s*[GMK]?Hz\s*-\s*([\d\.]+\s*[GMK]?Hz)', re.IGNORECASE),
        re.compile(r'current CPU frequency:\s*([\d\.]+\s*[GMK]?Hz)', re.IGNORECASE)
    )

    def check(self):
        try:
            output = subprocess.check_output(
                self.CPUPOWER_CMD,
                stderr=subprocess.DEVNULL,
                text=True
            )
        except Exception:
            return False

        limit_pattern, cur_pattern = self.CPUPOWER_PATTERNS
        max_limit_match = limit_pattern.search(output)
        cur_freq_match = cur_pattern.search(output)

        if max_limit_match and cur_freq_match:
            max_limit = max_limit_match.group(1).strip()
            cur_freq = cur_freq_match.group(1).strip()
            return max_limit == cur_freq
        return False


class LshwStrategy(CPUHighPerformanceStrategy):
    LSHW_CMD = shlex.split("lshw -c cpu")
    LSHW_PATTERNS = (
        re.compile(r'size:\s*([^\n]+)', re.IGNORECASE),
        re.compile(r'capacity:\s*([^\n]+)', re.IGNORECASE)
    )

    def check(self):
        try:
            lshw_output = subprocess.check_output(
                self.LSHW_CMD,
                stderr=subprocess.DEVNULL,
                text=True
            )
        except Exception:
            return False

        sizes = []
        capacities = []

        size_pattern, capacity_pattern = self.LSHW_PATTERNS
        for line in lshw_output.splitlines():
            m_size = size_pattern.search(line)
            m_capacity = capacity_pattern.search(line)
            if m_size:
                sizes.append(m_size.group(1).strip())
            if m_capacity:
                capacities.append(m_capacity.group(1).strip())

        return bool(sizes and capacities and sizes == capacities)


class ScalingGovernorStrategy(CPUHighPerformanceStrategy):
    SCALING_GOVERNOR_PATH = '/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor'

    def check(self):
        cpu_count = os.cpu_count()
        if cpu_count is None:
            return False

        for core_id in range(cpu_count):
            gov_path = self.SCALING_GOVERNOR_PATH.format(core_id)
            try:
                with open_s(gov_path, 'r', encoding='utf-8') as f:
                    if f.read().strip() != "performance":
                        return False
            except Exception:
                return False
        return True


class CPUHighPerformanceCollector(BaseCollector):
    strategies = [
        DmidecodeStrategy(),
        ScalingGovernorStrategy(),
        CpupowerStrategy(),
        PsutilStrategy(),
        LshwStrategy()
    ]

    def __init__(self, error_handler=None, *, strategies=None):
        super().__init__(error_handler)
        if strategies:
            self.strategies = strategies

    def _collect_data(self):
        high_performance = any(strategy.check() for strategy in self.strategies)
        return {"high_performance": high_performance}


# *** Kernel Info ***
class KernelInfoCollector(BaseCollector):
    TRANSPARENT_HUGEPAGE_PATH = '/sys/kernel/mm/transparent_hugepage/enabled'

    def _collect_data(self):
        kernel_info = platform.uname()._asdict()
        try:
            with open_s(self.TRANSPARENT_HUGEPAGE_PATH) as f:
                content = f.read()
        except Exception as e:
            self.error_handler.add_error(
                filename=__file__,
                function='_collect_data',
                lineno=265,
                what=f"打开文件失败：{self.TRANSPARENT_HUGEPAGE_PATH!r}",
                reason=str(e)
            )
            content = None

        if content:
            m = re.search(r'\[(\w+)\]', content)
            kernel_info['transparent_hugepage'] = m.group(1) if m else content.strip()
        return kernel_info


# *** Memory Info ***
class MemoryInfoCollector(BaseCollector):
    OVERCOMMIT_MEMORY_PATH = '/proc/sys/vm/overcommit_memory'

    def _collect_data(self):
        mem_info = {}
        try:
            mem_info['page_size'] = os.sysconf("SC_PAGESIZE")
        except Exception as e:
            self.error_handler.add_error(
                filename=__file__,
                function='_collect_data',
                lineno=289,
                what="获取 PAGESIZE 失败：os.sysconf('SC_PAGESIZE')",
                reason=str(e)
            )

        try:
            with open_s(self.OVERCOMMIT_MEMORY_PATH) as f:
                mem_info['overcommit_memory'] = f.read().strip()
        except Exception as e:
            self.error_handler.add_error(
                filename=__file__,
                function='_collect_data',
                lineno=301,
                what=f"打开文件失败：{self.OVERCOMMIT_MEMORY_PATH!r}",
                reason=str(e)
            )
        
        return mem_info


class SysCollector(BaseCollector):
    subcollectors = [
        LscpuCollector(),
        VirtualMachineCollector(),
        CPUHighPerformanceCollector(),
        KernelInfoCollector(),
        MemoryInfoCollector()
    ]

    def __init__(self, error_handler=None, *, subcollectors=None):
        super().__init__(error_handler)
        self.error_handler.type = "system"
        if subcollectors:
            self.subcollectors = subcollectors

    def _collect_data(self): 
        max_workers = min(len(self.subcollectors), os.cpu_count() or 1)
        ret = {}

        with ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(collector.collect) for collector in self.subcollectors]
            for future in as_completed(futures):
                result = future.result()
                self.error_handler.extend(result.error_handler)
                ret.update(result.data)

        return ret
