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
import torch

from msit_llm.common.log import logger
from msit_llm.dump.torch_dump.dump_config import DumpConfig
from msit_llm.dump.torch_dump.dump_hook import DumpHookModule


HOOK_TYPE = ""


def get_device(model):
    device = "cpu"

    for param in model.parameters():
        device = str(param.device)

    return device.replace(":", "")


def register_hook(model, config: DumpConfig, hook_type="dump_data"):
    global HOOK_TYPE
    if HOOK_TYPE:  # 避免重复hook
        logger.warning("%s has been register in model.", HOOK_TYPE)
        return
    else:
        HOOK_TYPE = hook_type

    if not isinstance(model, torch.nn.Module):
        logger.error("model must be instance of torch.nn.Module.")
        return

    if type(config).__name__ != "DumpConfig":
        logger.error("config must be instance of DumpConfig.")
        return

    device = get_device(model)
    config.set_dump_device_and_dump_dir(device)

    if HOOK_TYPE == "dump_data":
        hook_module = DumpHookModule(model, config)
        hook_module.add_hook()
