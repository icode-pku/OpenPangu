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
import shutil
import re
import pandas as pd

from components.utils.file_open_check import FileStat
from components.utils.file_open_check import ms_open
from components.utils.constants import TENSOR_MAX_SIZE
from components.utils.security_check import ms_makedirs
from msit_llm.common.log import logger
from msit_llm.common.utils import safe_string, load_file_to_read_common_check
from msit_llm.common.constant import ATB_HOME_PATH, ATB_SAVE_TENSOR_TIME, ATB_SAVE_TENSOR_IDS, \
    ATB_SAVE_TENSOR_RUNNER, ATB_SAVE_TENSOR, ATB_SAVE_TENSOR_RANGE, \
    ATB_SAVE_TILING, LD_PRELOAD, ATB_OUTPUT_DIR, ATB_SAVE_CHILD, ATB_SAVE_TENSOR_PART, \
    ASCEND_TOOLKIT_HOME, ATB_PROB_LIB_WITH_ABI, ATB_PROB_LIB_WITHOUT_ABI, ATB_SAVE_CPU_PROFILING, \
    ATB_CUR_PID, ATB_DUMP_SUB_PROC_INFO_SAVE_PATH, ATB_DEVICE_ID, ATB_AIT_LOG_LEVEL, ATB_DUMP_TYPE, get_ait_dump_path, \
    ATB_TIMESTAMP, GLOBAL_HISTORY_AIT_DUMP_PATH_LIST, ATB_SAVE_TENSOR_IN_BEFORE_OUT_AFTER, ATB_SAVE_TENSOR_STATISTICS, \
    ATB_SAVE_SYMLINK, ATB_DUMP_CONFIG


def run_pipeline(lib_atb_path):
    nm_process = subprocess.run(
        ['/usr/bin/nm', '-D', lib_atb_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    grep1_process = subprocess.run(
        ['/usr/bin/grep', 'Probe'],
        input=nm_process.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    grep2_process = subprocess.run(
        ['/usr/bin/grep', 'cxx11'],
        input=grep1_process.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    total_output = (
        nm_process.stderr + 
        grep1_process.stderr + 
        grep2_process.stderr + 
        grep2_process.stdout
    ).decode('utf-8', errors='ignore')
    result_code = grep2_process.returncode
    
    return result_code, total_output


def is_use_cxx11():
    atb_home_path = os.environ.get(ATB_HOME_PATH, "")
    if not atb_home_path or not os.path.exists(atb_home_path):
        raise OSError("ATB_HOME_PATH from atb is required, but it is empty or invalid.")
    atb_home_path = safe_string(atb_home_path)
    lib_atb_path = os.path.join(atb_home_path, "lib", "libatb.so")
    if not os.path.exists(lib_atb_path):
        raise OSError(f"{lib_atb_path} not exists, please make sure atb is compiled correctly")

    result_code, abi_result = run_pipeline(lib_atb_path)
    if result_code == 1 and len(abi_result) == 0:  # Execute succesfully but not found
        return False
    elif result_code != 0:
        logger.warning("Detecting abi status from atb so failed, will regard it as False")
        return False
    else:
        return len(abi_result) > 0


def init_dump_task(args):
    if args.config_path:
        os.environ[ATB_DUMP_CONFIG] = str(args.config_path)

    if args.save_desc:
        os.environ[ATB_SAVE_TENSOR] = "2"
    else:
        os.environ[ATB_SAVE_TENSOR] = "1"

    if args.time == 3:
        os.environ[ATB_SAVE_TENSOR_IN_BEFORE_OUT_AFTER] = "1"
        os.environ[ATB_SAVE_TENSOR_TIME] = "1"
    else:
        os.environ[ATB_SAVE_TENSOR_IN_BEFORE_OUT_AFTER] = "0"
        os.environ[ATB_SAVE_TENSOR_TIME] = str(args.time)

    if args.ids:
        os.environ[ATB_SAVE_TENSOR_IDS] = str(args.ids)
    else:
        os.environ.pop(ATB_SAVE_TENSOR_IDS, None)  # Ensure none is set

    if args.opname:
        os.environ[ATB_SAVE_TENSOR_RUNNER] = str(args.opname).lower()
    else:
        os.environ.pop(ATB_SAVE_TENSOR_RUNNER, None)  # Ensure none is set

    get_ait_dump_path()

    if args.output:
        if os.path.abspath(str(args.output)).endswith('/'):
            os.environ[ATB_OUTPUT_DIR] = os.path.abspath(str(args.output))
        else:
            os.environ[ATB_OUTPUT_DIR] = os.path.abspath(str(args.output)) + '/'
    else:
        os.environ.pop(ATB_OUTPUT_DIR, None)  # Ensure none is set

    if "onnx" in args.type:
        args.type.append("model")
        args.type.append("op")
        
    if args.type:
        os.environ[ATB_DUMP_TYPE] = "|".join(args.type)
    else:
        os.environ.pop(ATB_DUMP_TYPE, None)  # Ensure none is set

    if "stats" in args.type and "tensor" in args.type:
        os.environ[ATB_SAVE_TENSOR_STATISTICS] = "1"
    else:
        os.environ.pop(ATB_SAVE_TENSOR_STATISTICS, None)  # Ensure none is set

    if "onnx" in args.type and ("model" in args.type or "layer" in args.type):
        os.environ[ATB_DUMP_SUB_PROC_INFO_SAVE_PATH] = os.path.join(str(args.output), str(os.getpid()))
        subprocess_info_path = os.path.join(args.output, str(os.getpid()))
        ms_makedirs(subprocess_info_path, exist_ok=True)
    else:
        os.environ.pop(ATB_DUMP_SUB_PROC_INFO_SAVE_PATH, None)  # Ensure none is set

    if args.set_random_seed is not None:
        from msit_llm import seed_all
        seed = int(args.set_random_seed)
        seed_all(seed=seed)

    os.environ[ATB_SAVE_CHILD] = "1" if args.child else "0"
    os.environ[ATB_SAVE_TENSOR_RANGE] = str(args.range)
    os.environ[ATB_SAVE_TILING] = "1" if args.tiling else "0"
    os.environ[ATB_SAVE_TENSOR_PART] = str(args.save_tensor_part)
    os.environ[ATB_SAVE_CPU_PROFILING] = "1" if "cpu_profiling" in args.type else "0"
    os.environ[ATB_SAVE_SYMLINK] = "1" if args.enable_symlink else "0"
    os.environ[ATB_CUR_PID] = str(os.getpid())

    if args.device_id is not None:
        os.environ[ATB_DEVICE_ID] = str(args.device_id)
    else:
        os.environ.pop(ATB_DEVICE_ID, None)  # Ensure none is set

    atb_log_level_map = {"debug": 0, "info": 1, "warning": 2, "warn": 2, "error": 3, "fatal": 4, "critical": 5}
    cur_log_level = atb_log_level_map.get(args.log_level.lower(), 1)
    os.environ[ATB_AIT_LOG_LEVEL] = str(cur_log_level)

    cann_path = os.environ.get(ASCEND_TOOLKIT_HOME, "/usr/local/Ascend/ascend-toolkit/latest")
    if not cann_path or not os.path.exists(cann_path):
        raise OSError("cann_path is invalid, please install cann-toolkit and set the environment variables.")

    cur_is_use_cxx11 = is_use_cxx11()
    logger.info(f"Info detected from ATB so is_use_cxx11: {cur_is_use_cxx11}")
    save_tensor_so_name = ATB_PROB_LIB_WITH_ABI if cur_is_use_cxx11 else ATB_PROB_LIB_WITHOUT_ABI
    save_tensor_so_path = os.path.join(cann_path, "tools", "ait_backend", "dump", save_tensor_so_name)
    if not os.path.exists(save_tensor_so_path):
        raise OSError(f"{save_tensor_so_name} is not found in {cann_path}. Try installing the latest cann-toolkit")
    if not FileStat(save_tensor_so_path).is_basically_legal('read', strict_permission=True):
        raise OSError(f"{save_tensor_so_name} is illegal, group or others writable file stat is not permitted")

    logger.info(f"Append save_tensor_so_path: {save_tensor_so_path} to LD_PRELOAD")
    ld_preload = os.getenv(LD_PRELOAD)
    if ld_preload:
        os.environ[LD_PRELOAD] = save_tensor_so_path + ":" + ld_preload
    else:
        os.environ[LD_PRELOAD] = save_tensor_so_path


def json_to_onnx(args):
    subprocess_info_file = os.path.join(str(args.output), str(os.getpid()), 'subprocess_info.txt')
    if not os.path.exists(subprocess_info_file):
        return

    subprocess_info_file = load_file_to_read_common_check(subprocess_info_file)
    with ms_open(subprocess_info_file, max_size=TENSOR_MAX_SIZE) as f:
        from msit_llm.common.json_fitter import atb_json_to_onnx
        cache_csv_file = {}
        for line in f.readlines():
            path = line.strip()
            if not os.path.exists(path):
                continue
            atb_json_to_onnx(path, cache_csv_file=cache_csv_file)

    # clean tmp file
    subprocess_info_dir = os.path.join(args.output, str(os.getpid()))
    if os.path.isdir(subprocess_info_dir):
        shutil.rmtree(subprocess_info_dir)


def read_cpu_profiling_data(lines, data_map):
    for line in lines:
        # 解析opname和数据
        match = re.match(r'\[([a-zA-Z0-9_]*)\]:(.*)', line)
        if match:
            opname = match.group(1)
            stats = match.group(2)
            # 将数据添加到字典中
            if opname in data_map:
                data_map[opname].append(stats)
            else:
                data_map[opname] = [stats]


def split_cpu_profiling_data(data, opname):
    execute_data = ''
    setup_data = ''
    # 遍历每个opname的数据
    for stats in data[opname]:
        # 提取execute和setup数据
        execute_match = re.search(r'kernelExecuteTime:(\d+)', stats)
        setup_match = re.search(r'runnerSetupTime:(\d+)', stats)
        if execute_match:
            execute_data = stats
        elif setup_match:
            setup_data = stats
    return execute_data, setup_data


def add_extracted_headers_to_csv(split_data: str, csv_buffer: dict, prefix: str, headers: set):
    if not split_data:
        return
    matched_pairs = split_data.split(", ")
    for header_value_pair in matched_pairs:
        header, value = header_value_pair.split(':')
        headers.add(prefix + header)
        csv_buffer[prefix + header] = value


def merge_cpu_profiling_data(path):
    # 遍历目录下所有文件
    for root, _, files in os.walk(path):
        for file in files:
            if not re.match(r'operation_statistic_\d+\.txt', file):
                continue
            data = {}
            headers = set()
            csv_buffer_data = list()
            
            file_path = load_file_to_read_common_check(os.path.join(root, file))
            with ms_open(file_path, 'r', max_size=TENSOR_MAX_SIZE) as f:
                lines = f.readlines()
                read_cpu_profiling_data(lines, data)
                for opname in data.keys():
                    execute_data, setup_data = split_cpu_profiling_data(data, opname)
                    # 按照header类型添加prefix标识
                    csv_buffer_data.append({'opname': opname})
                    add_extracted_headers_to_csv(execute_data, csv_buffer_data[-1], 'execute_', headers)
                    add_extracted_headers_to_csv(setup_data, csv_buffer_data[-1], 'setup_', headers)

            headers = sorted(list(headers))
            headers.insert(0, 'opname')
            csv_dump_path = os.path.join(root, os.path.splitext(file)[0]) + '.csv'
            pd.DataFrame(csv_buffer_data).fillna('').to_csv(csv_dump_path, columns=headers, index=False)
            os.chmod(csv_dump_path, 0o640)
            
            # 删除原始文件
            os.remove(os.path.join(root, file))


def clear_dump_task(args):
    if "onnx" in args.type and ("model" in args.type or "layer" in args.type):
        json_to_onnx(args)
    if "cpu_profiling" in args.type:
        # 获取当前进程的cpu_profiling数据dump路径，新版CANN包需要加时间戳，否则不加时间戳
        for x in GLOBAL_HISTORY_AIT_DUMP_PATH_LIST:
            atb_output_dir = os.environ.get(ATB_OUTPUT_DIR, "")
            timestamp = os.environ.get(ATB_TIMESTAMP, "")
            cpu_profiling_path1 = os.path.join(atb_output_dir, "_".join([x, timestamp]), "cpu_profiling")
            cpu_profiling_path2 = os.path.join(atb_output_dir, x, "cpu_profiling")
            if os.path.exists(cpu_profiling_path1):
                merge_cpu_profiling_data(cpu_profiling_path1)
                return
            elif os.path.exists(cpu_profiling_path2):
                merge_cpu_profiling_data(cpu_profiling_path2)
                return
    return
