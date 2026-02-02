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

import os
import subprocess

from components.utils.file_open_check import FileStat
from msit_llm.common.constant import LD_PRELOAD, ATB_PROB_LIB_WITH_ABI, ATB_PROB_LIB_WITHOUT_ABI, \
                                ASCEND_TOOLKIT_HOME, ATB_OUTPUT_DIR, ATB_CHECK_TYPE, CHECK_TYPE_MAPPING, \
                                ATB_EXIT, ATB_AIT_LOG_LEVEL
from msit_llm.common.log import logger
from msit_llm.dump.initial import is_use_cxx11
from components.utils.util import filter_cmd
            

def handles_check_type(args) -> None:
    """Set error check type for backend usage from user input."""
    os.environ[ATB_CHECK_TYPE] = ''.join(CHECK_TYPE_MAPPING[type_] for type_ in args.type)


def handles_output_dir(args) -> None:
    """Set output directory for backend usage from user input."""
    output_dir = args.output

    if not output_dir:
        output_dir = os.getcwd()
    else:
        output_dir = os.path.realpath(output_dir)

    os.environ[ATB_OUTPUT_DIR] = output_dir
    

def handles_exit_flag(args) -> None:
    """Set exiting flag for backend usage from user input"""
    os.environ[ATB_EXIT] = '1' if args.exit else '0'


def handles_so_dir() -> None:
    """locate and check the dependent libaries."""
    cann_path = os.environ.get(ASCEND_TOOLKIT_HOME, "/usr/local/Ascend/ascend-toolkit/latest")
    
    if not cann_path or not os.path.exists(cann_path):
        raise OSError("CANN path is invalid, please install ascend toolkit and set the environment variables.")

    cur_is_use_cxx11 = is_use_cxx11()
    save_tensor_so_name = ATB_PROB_LIB_WITH_ABI if cur_is_use_cxx11 else ATB_PROB_LIB_WITHOUT_ABI

    save_tensor_so_path = os.path.join(cann_path, "tools", "ait_backend", "dump", save_tensor_so_name)
    so_real_path = os.path.realpath(save_tensor_so_path)

    if not os.path.exists(so_real_path):
        raise OSError(f"{save_tensor_so_name} is not found. Try to install the latest ascend toolkit")
    
    if not FileStat(so_real_path).is_basically_legal('read', strict_permission=True):
        raise OSError(f"{save_tensor_so_name} is illegal, group or others writable file stat is not permitted")

    ld_preload = os.getenv(LD_PRELOAD)
    if ld_preload:
        os.environ[LD_PRELOAD] = so_real_path + ":" + ld_preload
    else:
        os.environ[LD_PRELOAD] = so_real_path


def handles_exec(args) -> None:
    """handles executable subcommand from user input."""
    
    if not args.exec or args.exec.isspace():
        raise ValueError("exec expected executable subcommand, got empty instead")
    
    logger.info("Preparing to execute the command: %s", args.exec)
    logger.warning("Please make sure that the executable command is safe.")
    
    cmds = args.exec.split()
    cmds = filter_cmd(cmds)
    subprocess.run(cmds, shell=False)


def process_error_check(args) -> None:
    atb_log_level_map = {
        "debug": '0', "info": '1', "warning": '2', "warn": '2', "error": '3', "fatal": '4', "critical": '5'
    }

    os.environ[ATB_AIT_LOG_LEVEL] = atb_log_level_map.get(args.log_level, 1)

    logger.info("Environment configuring...")

    logger.info("User inputs verifying...")
    handles_check_type(args)
    handles_output_dir(args)
    handles_exit_flag(args)
    logger.info("User inputs verified.")
    
    logger.info("Dependencies checking...")
    handles_so_dir()
    logger.info("Dependencies founded.")
    
    logger.info("Environment configuration finished. Inference processing...") 
    handles_exec(args)
    logger.info("Inference finished.")
    
