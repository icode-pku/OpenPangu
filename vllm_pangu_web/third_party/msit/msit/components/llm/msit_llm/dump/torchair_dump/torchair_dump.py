# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
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

from components.utils.constants import FileCheckConst
from components.utils.file_utils import (
    check_and_get_real_path,
    create_directory,
    check_path_readability,
    check_path_writability,
    check_path_type
)
from msit_llm.common.constant import TORCHAIR_FX_DATA_DIRECTORY, TORCHAIR_GE_DATA_DIRECTORY
from msit_llm.common.log import logger


def try_import_torchair():
    try:
        import torch
        import torch_npu
        import torchair
    except ModuleNotFoundError as e:
        logger.error("torch or torch_npu with torchair not found. Try installing the latest torch_npu.")
        raise e


def get_ge_dump_config(
        dump_path='',
        dump_mode='all',
        fusion_switch_file=None,
        dump_token=None,
        dump_layer=None,
        compiler_config=None
):
    if not isinstance(dump_path, str):
        raise TypeError(f"dump_path should be a string, but got a {type(dump_path)}")
    dump_path = os.path.join(dump_path, TORCHAIR_GE_DATA_DIRECTORY)
    dump_path = check_and_get_real_path(dump_path, FileCheckConst.READ_WRITE_ABLE, must_exist=False)
    create_directory(dump_path)
    check_path_readability(dump_path)
    check_path_writability(dump_path)
    if os.listdir(dump_path):
        logger.warning('The specified directory for GE-Dump data is not empty, which may result in data mixing.')

    try_import_torchair()
    from torchair.configs.compiler_config import CompilerConfig

    if compiler_config is not None and not isinstance(compiler_config, CompilerConfig):
        raise TypeError(f'compiler_config must be a CompilerConfig, but got {type(compiler_config)}')
    config = compiler_config if compiler_config else CompilerConfig()
    # Generate GE mapping graph
    config.debug.graph_dump.type = "txt"
    if hasattr(config.debug.graph_dump, "_path"):  # interface changed since 8.0.RC1.b080
        setattr(config.debug.graph_dump, "_path", dump_path)
    else:
        config.debug.graph_dump.path = dump_path

    if fusion_switch_file is not None:
        check_path_type(fusion_switch_file, FileCheckConst.FILE)
        fusion_switch_file = check_and_get_real_path(fusion_switch_file, FileCheckConst.READ_ABLE)
        config.fusion_config.fusion_switch_file = fusion_switch_file

    # Enable GE dump
    config.dump_config.enable_dump = True
    config.dump_config.dump_mode = dump_mode
    config.dump_config.dump_path = dump_path

    if dump_token is not None:
        new_token = [str(x) for x in dump_token]
        config.dump_config.dump_step = "|".join(new_token)
    if dump_layer is not None:
        dump_layer = " ".join(dump_layer)
        config.dump_config.dump_layer = dump_layer

    return config


def get_fx_dump_config(dump_path='', compiler_config=None):
    if not isinstance(dump_path, str):
        raise TypeError(f"dump_path should be a string, but got a {type(dump_path)}")
    dump_path = os.path.join(dump_path, TORCHAIR_FX_DATA_DIRECTORY)
    dump_path = check_and_get_real_path(dump_path, FileCheckConst.READ_WRITE_ABLE, must_exist=False)
    create_directory(dump_path)
    check_path_readability(dump_path)
    check_path_writability(dump_path)
    if os.listdir(dump_path):
        logger.warning('The specified directory for FX-Dump data is not empty, which may result in data mixing.')

    try_import_torchair()
    from torchair.configs.compiler_config import CompilerConfig

    if compiler_config is not None and not isinstance(compiler_config, CompilerConfig):
        raise TypeError(f'compiler_config must be a CompilerConfig, but got {type(compiler_config)}')
    config = compiler_config if compiler_config else CompilerConfig()
    # Enable FX dump
    config.debug.data_dump.type = "npy"
    if hasattr(config.debug.data_dump, "path"):
        config.debug.data_dump.path = dump_path

    return config
