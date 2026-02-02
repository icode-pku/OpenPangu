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

import re
import os
import stat
import json
import shlex
import itertools
import subprocess
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Type

from msguard.security import open_s
from .log import global_logger


class NpuType(Enum):
    TP_300 = "d100"
    TP_300I_DUO = "d500"
    TP_A1 = "d801"
    TP_A2 = "d802"
    TP_A3 = "d803"

    @property
    def display(self):
        return {
            NpuType.TP_300: "300",
            NpuType.TP_300I_DUO: "300I Duo",
            NpuType.TP_A1: "800I A1",
            NpuType.TP_A2: "800I A2",
            NpuType.TP_A3: "800I A3",
        }[self]


def get_npu_count():
    davinci_path_template = "/dev/davinci{}"

    for device_id in itertools.count(0):
        device_path = davinci_path_template.format(device_id)
        try:
            f_mode = os.stat(device_path).st_mode
        except Exception:
            break

        if not stat.S_ISCHR(f_mode):
            break

    return device_id


def get_npu_type():
    try:
        output = subprocess.check_output(['/usr/bin/lspci'], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None, None

    device_pattern = re.compile(r'device\s*(\w+)', re.IGNORECASE)
    device_list = []
    zip_engine_nums = 0
    for line in output.split('\n'):
        if 'accelerator' in line:
            m = device_pattern.search(line)
            if not m:
                zip_engine_nums += 1 # d803 contains extra zip engines
                continue
            device_list.append(m.group(1))
    
    if not device_list:
        return None, None

    if any(device != device_list[0] for device in device_list):
        return None, None
    
    npu_type = device_list[0]
    if npu_type not in (member.value for member in NpuType):
        return None, None

    return NpuType(npu_type), len(device_list) - zip_engine_nums


def get_conn_mode():
    lldp_cmd = "hccn_tool -i 0 -lldp -g"
    try:
        output = subprocess.check_output(shlex.split(lldp_cmd), stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None
    
    if not output:
        return None
    
    fields = output.split('\n')
    tlv_desc = 'System Description TLV'
    if tlv_desc not in fields:
        return None
    
    desc_idx = fields.index(tlv_desc)
    if desc_idx >= len(fields):
        return None
    
    route_fields = "Routing"
    fiber_fields = "AscendNPU"
    if route_fields in fields[desc_idx + 1]:
        return "route"
    
    if fiber_fields in fields[desc_idx + 1]:
        return "fiber"
    
    return None


# --- rank table ---
class FrameworkType(Enum):
    TP_MINDIE = "mindie"
    TP_VLLM = "vllm"


@dataclass
class DeviceInfo:
    device_ip: str
    device_id: int
    rank_id: int


@dataclass
class RankTable:
    host_to_devices: Dict[str, List[DeviceInfo]]
    server_count: int
    version: str


class ParserRegistry:
    """Rank Table Parser Registry. Dynamically register and get rank table parser."""
    _registry = {}

    @classmethod
    def register(cls, framework: FrameworkType):
        """Register a parser using decorator. Example usage: @ParserRegistry.register(xx)"""
        def wrapper(parser_cls):
            cls._registry[framework] = parser_cls
            return parser_cls
        return wrapper
    
    @classmethod
    def get(cls, framework: FrameworkType) -> Type['RankTableParser']:
        """Get a RankTableParser"""
        if framework not in cls._registry:
            raise ValueError(f"Not registered framework: {framework}. Registered framework: {set(cls._registry)}")
        
        return cls._registry[framework]


class RankTableParser(ABC):
    def __init__(self):
        single_address = "(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])"
        self._ip_pattern = re.compile(
            rf"\b{single_address}(?:\.{single_address}){{3}}\b"
        )
        self.host_limits = 1000 # max 1000 hosts in rank table
        self.device_limits_per_host = 32 # max 32 devices per host

    @abstractmethod
    def parse(self, rank_table_path: str) -> RankTable:
        pass


class JsonParser(RankTableParser):
    @staticmethod
    def _load_json(path: str):
        try:
            with open_s(path, 'r', encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            global_logger.warning("Error occured while loading json data: %s", e)
            return {}
    
    @abstractmethod
    def _parse_devices(self, data: dict) -> Dict[str, DeviceInfo]:
        pass

    def parse(self, rank_table_path: str):
        data = self._load_json(rank_table_path)
        if not data:
            return RankTable([], 0, "'")

        server_count = data.get('server_count')

        if not server_count or not server_count.isdigit():
            global_logger.warning("Expected 'server_count' to be a digit str. Got %r instead.", server_count)
            server_count = "0"
        else:
            server_count = int(server_count)
            
        version = data.get('version', "1.0")
        host_to_devices = self._parse_devices(data)

        return RankTable(
            host_to_devices=host_to_devices,
            server_count=server_count,
            version=version
        )


@ParserRegistry.register(FrameworkType.TP_MINDIE)
class MindIEParser(JsonParser):
    def _parse_devices(self, data: dict) -> Dict[str, DeviceInfo]:
        host_to_devices = {}

        for host_num, server_info in enumerate(data.get('server_list', {})):
            if host_num > self.host_limits:
                raise RuntimeError("Number of hosts found in rank table exceeds the limit")

            host_ip = server_info.get('server_id', "")
            if not self._ip_pattern.match(host_ip):
                global_logger.warning(
                    "Invalid 'server_id' from rank table: %r", host_ip
                )
                continue

            if host_ip not in host_to_devices:
                host_to_devices[host_ip] = []
            
            for device_num, device_info in enumerate(server_info.get('device', {})):
                if device_num > self.device_limits_per_host:
                    raise RuntimeError(
                        f"Number of devices for host {host_ip!r} found in rank table exceeds the limit"
                    )

                device_ip = device_info.get('device_ip')
                if not device_ip or not self._ip_pattern.match(device_ip):
                    global_logger.warning(
                        "Invalid 'device_ip' for 'server_id' %r from rank table: %r",
                        device_ip, host_ip
                    )
                    continue
                
                device_id = device_info.get('device_id')
                if not device_id or not device_id.isdigit():
                    global_logger.warning(
                        "Expected 'device_id' for 'server_id' %r to be a digit str. Got %r instead.",
                        device_id, host_ip
                    )
                    continue
                device_id = int(device_id)

                rank_id = device_info.get('rank_id')
                if not rank_id or not rank_id.isdigit():
                    global_logger.warning(
                        "Expected 'rank_id' for host %r to be a digit str. Got %r instead.",
                        rank_id, host_ip
                    )
                    continue
                rank_id = int(rank_id)

                host_to_devices[host_ip].append(
                    DeviceInfo(device_ip=device_ip, device_id=device_id, rank_id=rank_id)
                )

        return host_to_devices


@ParserRegistry.register(FrameworkType.TP_VLLM)
class VLLMParser(JsonParser):
    def _parse_devices(self, data: dict) -> Dict[str, DeviceInfo]:
        host_to_devices = {}

        for device_list_name in ("prefill_device_list", "decode_device_list"):
            if device_list_name not in data:
                global_logger.warning(
                    "Expected %r in rank table, but not found.", device_list_name
                )
                continue

            if len(data[device_list_name]) > self.host_limits * self.device_limits_per_host:
                raise RuntimeError("Number of items found in rank table exceeds the limit")

            for device_info in data[device_list_name]:
                host_ip = device_info.get('server_id')
                if not host_ip or not self._ip_pattern.match(host_ip):
                    global_logger.warning(
                        "Invalid host_ip from rank table: %r", host_ip
                    )
                    continue
                
                if host_ip not in host_to_devices:
                    host_to_devices[host_ip] = []
                
                if len(host_to_devices) > self.host_limits:
                    raise RuntimeError("Number of hosts found in rank table exceeds the limit")

                if len(host_to_devices[host_ip]) > self.device_limits_per_host:
                    raise RuntimeError(
                        f"Number of devices for host {host_ip!r} found in rank table exceeds the limit"
                    )

                device_ip = device_info.get('device_ip')
                if not device_ip or not self._ip_pattern.match(device_ip):
                    global_logger.warning(
                        "Invalid 'device_ip' for 'server_id' %r from rank table: %r",
                        device_ip, host_ip
                    )
                    continue
        
                device_id = device_info.get('device_id')
                if not device_id or not device_id.isdigit():
                    global_logger.warning(
                        "Expected 'device_id' for 'server_id' %r to be a digit str. Got %r instead.",
                        device_id, host_ip
                    )
                    continue

                device_id = int(device_id)
                rank_id = device_info.get('cluster_id')
                if not rank_id or not rank_id.isdigit():
                    global_logger.warning(
                        "Expected 'cluster_id' for host %r to be a digit str. Got %r instead.",
                        rank_id, host_ip
                    )
                    continue

                rank_id = int(rank_id) - 1 # vllm cluster_id starts from 1
                host_to_devices[host_ip].append(
                    DeviceInfo(device_ip=device_ip, device_id=device_id, rank_id=rank_id)
                )

        return host_to_devices


model_type = None


def update_model_type(args):
    weight_dir = None
    if getattr(args, 'weight_dir', None):
        weight_dir = args.weight_dir
    elif getattr(args, 'mies_config_path', None):
        with open_s(args.mies_config_path) as f:
            data = json.load(f)
        try:
            weight_dir = data['BackendConfig']['ModelDeployConfig']['ModelConfig'][0]['modelWeightPath']
        except Exception:
            weight_dir = None

    if not weight_dir:
        return

    model_config_path = os.path.join(weight_dir, "config.json")

    global model_type
    try:
        with open_s(model_config_path) as f:
            data = json.load(f)
    except Exception:
        model_type = None
    else:
        model_type = data.get('model_type')


def get_model_type():
    return model_type
