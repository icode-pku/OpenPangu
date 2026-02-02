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
import functools
from dataclasses import dataclass
import os
import os.path

import torch

from msit_llm.dump.torch_dump.dump_config import DumpConfig
from msit_llm.dump.torch_dump import hook_ops
from msit_llm.common.log import logger
from components.utils.security_check import ms_makedirs
from msit_llm.dump.torch_dump.dump_statistics import dump_statistics


class DumpHookModule:
    def __init__(self, model, dump_config=None):
        self.model: torch.nn.Module = model
        self.dump_config = dump_config
        self.ori_torch_attr = {}
        hook_ops.add_torch_ops()
        hook_ops.add_torch_npu_ops()

    def add_hook(self):
        self._add_module_hook()
        self._add_api_hook()
        self._dump_module_weight()

    def remove_hook(self):
        self._remove_module_hook()
        self._remove_api_hook()

    def _add_module_hook(self):
        model_name = "root"

        def add_hook(module, prefix=""):
            module.ait_forward_handle = module.register_forward_hook(dump_module_data())
            module.ait_forward_pre_handle = module.register_forward_pre_hook(pre_forward_module())
            module.msit_name = prefix
            for name, child_module in module.named_children():
                add_hook(child_module, prefix + "." + name)

        self.model.msit_name = model_name
        self.model.ait_forward_pre_handle = self.model.register_forward_pre_hook(set_dump_flag())
        self.model.model_ait_forward_handle = self.model.register_forward_hook(dump_logits())
        add_hook(self.model, prefix=model_name)

    def _remove_module_hook(self):
        if hasattr(self.model, "ait_forward_pre_handle"):
            self.model.ait_forward_pre_handle.remove()

        if hasattr(self.model, "model_ait_forward_handle"):
            self.model.model_ait_forward_handle.remove()

        def _remove_hook(module):
            if hasattr(module, "ait_forward_handle"):
                module.ait_forward_handle.remove()
            if hasattr(module, "ait_forward_pre_handle"):
                module.ait_forward_pre_handle.remove()
            for _, _child_module in module.named_children():
                _remove_hook(_child_module)

        _remove_hook(self.model)

    def _add_api_hook(self):
        for py_module, api_list in hook_ops.HOOK_OPS.items():
            ori_module_attrs = {}
            for api_name in api_list:
                if not hasattr(py_module, api_name):
                    continue
                api = getattr(py_module, api_name)
                
                ori_module_attrs[api_name] = api_name
                new_api = wrap_torch_func(api, self.model)
                setattr(py_module, api_name, new_api)  # hook api

            self.ori_torch_attr[py_module] = ori_module_attrs

    def _remove_api_hook(self):
        for py_module, ori_attrs in self.ori_torch_attr:
            for api_name, api in ori_attrs:
                if not hasattr(py_module, api_name):
                    continue
                setattr(py_module, api_name, api)

    def _dump_module_weight(self):
        dump_config = DumpConfig()

        if not dump_config.dump_weight:
            return

        dump_path = os.path.join(dump_config.dump_dir, "weights")
        if not os.path.exists(dump_path):
            ms_makedirs(dump_path)

        prefix = "root"
        for name, paramter in self.model.named_parameters():
            dump_name = f"{prefix}.{name}" if name else prefix
            if not dump_config.is_dump_layer(dump_name):
                continue

            torch.save(paramter, os.path.join(dump_path, f"{dump_name}.pth"))


@dataclass
class DumpDataParams:
    inputs: any
    outputs: any
    dump_path: str
    tensor_part: int
    module_name: str
    dump_type: str

    
def wrap_torch_func(func, module: torch.nn.Module):

    @functools.wraps(func)
    def dump_api_data(*args, **kwargs):
        output = func(*args, **kwargs)

        dump_config = DumpConfig()
        if not dump_config.dump_flag or not dump_config.is_dump_cur_device or not dump_config.dump_api:
            return output
        dump_name = dump_config.get_api_folder_name(func.__name__)
        dump_type = module.type
        if not dump_config.is_dump_layer(dump_name, api=func):
            return output

        api_dump_path = os.path.join(dump_config.dump_dir, str(dump_config.token_id), dump_name)
        if not os.path.exists(api_dump_path):
            ms_makedirs(api_dump_path)
        params = DumpDataParams(
            inputs=args,
            outputs=output,
            dump_path=api_dump_path,
            tensor_part=dump_config.tensor_part,
            module_name=dump_name,
            dump_type=dump_type
        )


        # 然后将这个实例传递给dump_data函数：
        dump_data(params) 
        return output

    return dump_api_data


def dump_tensor(feat, feat_path, module_name, dump_type):
    dump_config = DumpConfig()
    if isinstance(feat, (tuple, list)):
        for idx, tensor in enumerate(feat):
            dump_tensor(tensor, "{}_{}".format(feat_path, idx), module_name, dump_type)
    elif isinstance(feat, torch.Tensor):
        if dump_config.dump_statistics_mode != 0:
            dump_statistics(feat, feat_path, module_name, dump_type)
        if dump_config.dump_statistics_mode != 1:
            if not feat_path.endswith(".pth"):
                feat_path += ".pth"
            torch.save(feat, feat_path)
        if dump_config.analyze is True:
            from msit_llm.dump.torch_dump.dump_analyze import dump_analyze
            dump_analyze(feat, feat_path, module_name, dump_type, dump_config.dump_path)
    else:
        logger.warning("Unrecognized data type %r, cannot be saved in path %r.", type(feat), feat_path)


def dump_data(params: DumpDataParams):
    if params.tensor_part == 0:
        dump_tensor(params.inputs, os.path.join(params.dump_path, "input"), params.module_name, params.dump_type)
    elif params.tensor_part == 1:
        dump_tensor(params.outputs, os.path.join(params.dump_path, "output"), params.module_name, params.dump_type)
    else:
        dump_tensor(params.inputs, os.path.join(params.dump_path, "input"), params.module_name, params.dump_type)
        dump_tensor(params.outputs, os.path.join(params.dump_path, "output"), params.module_name, params.dump_type)


def dump_module_data():
    exec_count = 0

    def hook_func(module: torch.nn.Module, inputs, outputs):
        nonlocal exec_count
        exec_count += 1
        dump_config = DumpConfig()
        module_name = module.msit_name
        dump_config.cur_module_name.pop()

        if not dump_config.is_dump_cur_device:
            return

        if dump_config.token_id == 0:
            dump_config.update_module_ids(module_name)
            # 将模型树状信息保存成json文件
            from msit_llm.dump.torch_dump.topo import ModelTree

            if not os.path.exists(dump_config.dump_dir):
                ms_makedirs(dump_config.dump_dir, mode=0o750)
            model_tree_path = os.path.join(dump_config.dump_dir, "model_tree.json")
            obj = ModelTree()
            obj.create_tree(module, dump_config.module_ids, model_tree_path)

        if dump_config.dump_last_logits:
            if has_tensor(outputs):
                dump_config.last_logits = (module_name, outputs)
            return

        if not dump_config.dump_module or not dump_config.dump_flag:
            return

        if not dump_config.is_dump_layer(module_name, module=module):
            return

        dump_path = os.path.join(dump_config.dump_dir, str(dump_config.token_id), module_name)
        dump_type = module.type
        if not os.path.exists(dump_path):
            ms_makedirs(dump_path, mode=0o750)
        # 使用时，创建一个DumpDataParams实例：
        params = DumpDataParams(
            inputs=inputs,
            outputs=outputs,
            dump_path=dump_path,
            tensor_part=dump_config.tensor_part,
            module_name=module_name,
            dump_type=dump_type
        )


        # 然后将这个实例传递给dump_data函数：
        dump_data(params)   


    return hook_func


def pre_forward_module():

    def hook_func(module: torch.nn.Module, _):
        dump_config = DumpConfig()
        module_name = module.msit_name
        dump_config.cur_module_name.append(module_name)

    return hook_func


def set_dump_flag():
    cur_token_id = 0

    def hook_func(module, inputs):
        nonlocal cur_token_id
        logger.debug("Current token id: %s", cur_token_id)
        config = DumpConfig()
        config.token_id = cur_token_id
        # 通过root module执行的轮次来判断当前在第几个token
        if config.token_id in config.token_range:
            config.dump_flag = True
        else:
            config.dump_flag = False

        cur_token_id += 1

    return hook_func


def has_tensor(feat):
    if isinstance(feat, (tuple, list)):
        for tensor in feat:
            if has_tensor(tensor):
                return True
    elif isinstance(feat, torch.Tensor):
        return True
    return False


def dump_logits():
    # 将缓存的输出dump到文件中
    def hook_func(module: torch.nn.Module, *args):
        config = DumpConfig()
        dump_type = module.type
        if config.dump_last_logits and config.last_logits is not None and config.dump_flag:
            module_name, outputs = config.last_logits

            dump_path = os.path.join(config.dump_dir, str(config.token_id), module_name)

            if not os.path.exists(dump_path):
                ms_makedirs(dump_path, mode=0o750)
            dump_tensor(outputs, os.path.join(dump_path, "output"), module_name, dump_type)

    return hook_func
