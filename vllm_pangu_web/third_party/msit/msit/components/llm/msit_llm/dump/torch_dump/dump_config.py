# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

from functools import wraps
from msit_llm.common import utils
from msit_llm.common.log import logger
from msit_llm.common.constant import get_ait_dump_path
from components.utils.security_check import ms_makedirs


def singleton(cls):
    ins = {}

    @wraps(cls)
    def run(*args, **kwargs):
        if cls not in ins:
            ins[cls] = cls(*args, **kwargs)
        return ins.get(cls)

    return run


def str_to_reg_str(name):
    # 仅支持 *: 匹配0到N个随意字符
    replace_name = (
        name.replace("\\", "\\\\")
        .replace(".", "\\.")
        .replace("?", "\\?")
        .replace("+", "\\+")
        .replace("^", "\\^")
        .replace("$", "\\$")
        .replace("|", "\\|")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("*", ".*")
    )
    return f'^{replace_name}$'


@singleton
class DumpConfig:
    def __init__(
            self,
            dump_path=None,
            token_range=None,
            analyze=False,
            module_list=None,
            api_list=None,
            tensor_part=2,
            device_id=None,
            dump_statistics_mode=0,
            dump_last_logits=False,
            mode=None,
            dump_weight=False,
            layer_name=None,
            seed=None,
    ):
        self.dump_path = dump_path or "./"
        self.mode = mode or ["module"]
        self.mode = self.mode if isinstance(self.mode, (list, tuple)) else [self.mode]
        self.dump_module = "module" in self.mode
        self.dump_api = "api" in self.mode
        self.token_range = token_range or [0]
        self.analyze = analyze
        self.module_list = module_list or []
        self.api_list = api_list or []
        self.tensor_part = tensor_part
        self.device_id = device_id
        self.is_dump_cur_device = True
        self.dump_flag = True
        self.token_id = 0
        self.module_ids = {}
        self.cur_module_id = 0
        self.dump_dir = ""
        self.dump_statistics_mode = dump_statistics_mode
        self.dump_last_logits = dump_last_logits
        self.last_logits = None
        self.dump_weight = dump_weight
        self.layer_name_reg = str_to_reg_str(layer_name) if layer_name is not None else ""  # 支持 *: 匹配0到N个随意支付
        self.cur_module_name = ["root"]
        self.api_folder_name_set = set()

        if not self._check_args():
            raise ValueError("Invalid args of DumpConfig.")

        if seed is not None:
            from msit_llm import seed_all
            seed_all(seed=seed)

    def set_dump_device_and_dump_dir(self, device):
        if self.device_id is not None and device != "cpu":
            # Get the first position of a digit char, and cut out like cuda0 -> cuda, npu12 -> npu
            device_type = device[: max(enumerate(device), key=lambda xx: str.isdigit(xx[1]))[0]]
            device_id_str = f"{device_type}{self.device_id}"  # -> npu0
            if device != device_id_str:
                self.is_dump_cur_device = False
                return

        cur_dump_path = "{}_{}".format(device, os.getpid())
        self.dump_dir = os.path.join(self.dump_path, get_ait_dump_path(), "torch_tensors", cur_dump_path)
        if not os.path.exists(self.dump_dir):
            ms_makedirs(self.dump_dir, mode=0o750)

    def update_module_ids(self, module_name):
        self.cur_module_id += 1
        if module_name not in self.module_ids:
            self.module_ids[module_name] = self.cur_module_id

    def is_dump_layer(self, name, module=None, api=None):
        if not re.match(self.layer_name_reg, name):
            return False

        if self.module_list and module and not isinstance(module, tuple(self.module_list)):
            return False

        if self.api_list and api and api not in self.api_list:
            return False

        return True

    def get_api_folder_name(self, api_name, index=0):
        if index == 0:
            folder_name = f"{self.cur_module_name[-1]}.{api_name}"
        else:
            folder_name = f"{self.cur_module_name[-1]}.{api_name}.{index}"
        if folder_name in self.api_folder_name_set:
            return self.get_api_folder_name(api_name, index + 1)
        else:
            return folder_name

    def _check_args(self):
        utils.check_output_path_legality(self.dump_path)
        if not isinstance(self.token_range, list):
            logger.error("The type of token_range must be list, please check.")
            return False
        if not isinstance(self.module_list, list):
            logger.error("The type of module_list must be list, please check.")
            return False
        if not isinstance(self.tensor_part, int):
            logger.error("The type of tensor_part must be int, please check.")
            return False
        if self.device_id is not None and not isinstance(self.device_id, int):
            logger.error("The type of device_id must be int, please check.")
            return False
        if self.tensor_part not in [0, 1, 2]:
            logger.error("tensor_part must be 0 or 1 or 2.")
            return False
        if self.dump_statistics_mode not in [0, 1, 2]:
            logger.error("dump_statistics_mode  must be 0 or 1 or 2.")
            return False
        return True
