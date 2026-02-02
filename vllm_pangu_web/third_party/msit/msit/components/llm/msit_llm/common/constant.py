# Copyright (c) 2023-2025 Huawei Technologies Co., Ltd.
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

import datetime
import os

import torch

from components.utils.cmp_algorithm import CMP_ALG_MAP
from components.utils.util import safe_int


def get_visible_device(envionment_name):
    device_str_id = os.environ.get(envionment_name, "0").split(",")[0]
    device_id = safe_int(device_str_id, envionment_name)
    return device_id


def get_global_device():
    if hasattr(torch, "npu") and torch.npu.is_available() and get_visible_device("ASCEND_VISIBLE_DEVICES") >= 0:
        return "npu"
    if hasattr(torch, "cuda") and torch.cuda.is_available() and get_visible_device("CUDA_VISIBLE_DEVICES") >= 0:
        return "cuda"
    return "cpu"


GLOBAL_AIT_DUMP_PATH = "msit_dump"
GLOBAL_HISTORY_AIT_DUMP_PATH_LIST = ["msit_dump", "ait_dump"]
GLOBAL_DEVICE = get_global_device()
DEVICE_DIST_MAP = {"cuda": "nccl", "npu": "hccl", "cpu": "gloo"}
GLOBAL_DIST_BACKEND = DEVICE_DIST_MAP.get(GLOBAL_DEVICE, "gloo")

ATB_HOME_PATH = "ATB_HOME_PATH"
ATB_DUMP_CONFIG = "ATB_DUMP_CONFIG"
ATB_CUR_PID = "ATB_CUR_PID"
ATB_SAVE_SYMLINK = "ATB_SAVE_SYMLINK"
ATB_SAVE_TENSOR_TIME = "ATB_SAVE_TENSOR_TIME"
ATB_SAVE_TENSOR_IN_BEFORE_OUT_AFTER = "ATB_SAVE_TENSOR_IN_BEFORE_OUT_AFTER"
ATB_SAVE_TENSOR_IDS = "ATB_SAVE_TENSOR_IDS"
ATB_SAVE_TENSOR_RUNNER = "ATB_SAVE_TENSOR_RUNNER"
ATB_SAVE_TENSOR = "ATB_SAVE_TENSOR"
ATB_SAVE_TENSOR_RANGE = "ATB_SAVE_TENSOR_RANGE"
ATB_SAVE_TENSOR_STATISTICS = "ATB_SAVE_TENSOR_STATISTICS"
ATB_SAVE_TILING = "ATB_SAVE_TILING"
ATB_OUTPUT_DIR = "ATB_OUTPUT_DIR"
ATB_SAVE_CHILD = "ATB_SAVE_CHILD"
ATB_SAVE_TENSOR_PART = "ATB_SAVE_TENSOR_PART"
ATB_SAVE_CPU_PROFILING = "ATB_SAVE_CPU_PROFILING"
ATB_DUMP_SUB_PROC_INFO_SAVE_PATH = "ATB_DUMP_SUB_PROC_INFO_SAVE_PATH"
ATB_DUMP_TYPE = "ATB_DUMP_TYPE"
ATB_DEVICE_ID = "ATB_DEVICE_ID"
ATB_AIT_LOG_LEVEL = "ATB_AIT_LOG_LEVEL"
ATB_TIMESTAMP = "ATB_TIMESTAMP"
RAW_INPUT_PATH = "RAW_INPUT_PATH"
LCCL_DETERMINISTIC = "LCCL_DETERMINISTIC"
HCCL_DETERMINISTIC = "HCCL_DETERMINISTIC"
ATB_MATMUL_SHUFFLE_K_ENABLE = "ATB_MATMUL_SHUFFLE_K_ENABLE"
ATB_LLM_LCOC_ENABLE = "ATB_LLM_LCOC_ENABLE"
PYTHON_HASH_SEED = "PYTHONHASHSEED"

# ERRORCHECK
ATB_CHECK_TYPE = "ATB_CHECK_TYPE"
CHECK_TYPE_MAPPING = {
    "overflow": "1",
}
ATB_EXIT = "ATB_EXIT"

LD_PRELOAD = "LD_PRELOAD"
LOG_TO_STDOUT = "LOG_TO_STDOUT"

ATTR_VERSION = "$Version"
ATTR_END = "$End"
ATTR_OBJECT_LENGTH = "$Object.Length"
ATTR_OBJECT_COUNT = "$Object.Count"
ATTR_OBJECT_PREFIX = "$Object."

MAX_DATA_SIZE = 2 * 1024 * 1024 * 1024  # 2G
MAX_WEIGHT_DATA_SIZE = 32 * 1024 * 1024 * 1024  # 32G

ASCEND_TOOLKIT_HOME = "ASCEND_TOOLKIT_HOME"
ATB_PROB_LIB_WITH_ABI = "libatb_probe_abi1.so"
ATB_PROB_LIB_WITHOUT_ABI = "libatb_probe_abi0.so"

DATA_ID = "data_id"
TOKEN_ID = "token_id"
MY_DATA_PATH = "my_data_path"
DTYPE = "dtype"
SHAPE = "shape"
MAX_VALUE = "max_value"
MIN_VALUE = "min_value"
MEAN_VALUE = "mean_value"
MY_DTYPE = "my_dtype"
MY_SHAPE = "my_shape"
MY_MAX_VALUE = "my_max_value"
MY_MIN_VALUE = "my_min_value"
MY_MEAN_VALUE = "my_mean_value"
GOLDEN_DATA_PATH = "golden_data_path"
GOLDEN_DTYPE = "golden_dtype"
GOLDEN_SHAPE = "golden_shape"
GOLDEN_MAX_VALUE = "golden_max_value"
GOLDEN_MIN_VALUE = "golden_min_value"
GOLDEN_MEAN_VALUE = "golden_mean_value"
CMP_FAIL_REASON = "cmp_fail_reason"
WEIGHT_NAME = "weight_name"
GOLDEN_OP_TYPE = "golden_op_type"
MY_OP_TYPE = "my_op_type"

MULTIPROCESS_TIMEOUT = 3600

CSV_GOLDEN_HEADER = [
    TOKEN_ID,
    DATA_ID,
    GOLDEN_DATA_PATH,
    GOLDEN_OP_TYPE,
    GOLDEN_DTYPE,
    GOLDEN_SHAPE,
    GOLDEN_MAX_VALUE,
    GOLDEN_MIN_VALUE,
    GOLDEN_MEAN_VALUE,
    MY_DATA_PATH,
    MY_OP_TYPE,
    MY_DTYPE,
    MY_SHAPE,
    MY_MAX_VALUE,
    MY_MIN_VALUE,
    MY_MEAN_VALUE,
]
CSV_GOLDEN_HEADER.extend(list(CMP_ALG_MAP.keys()))
CSV_GOLDEN_HEADER.append(CMP_FAIL_REASON)
CSV_CMP_WEIGHT_HEADER = [
    WEIGHT_NAME,
    GOLDEN_DTYPE,
    GOLDEN_SHAPE,
    GOLDEN_MAX_VALUE,
    GOLDEN_MIN_VALUE,
    GOLDEN_MEAN_VALUE,
    MY_DTYPE,
    MY_SHAPE,
    MY_MAX_VALUE,
    MY_MIN_VALUE,
    MY_MEAN_VALUE,
]
CSV_CMP_WEIGHT_HEADER.extend(list(CMP_ALG_MAP.keys()))

MSIT_BAD_CASE_FOLDER_NAME = 'msit_bad_case'

MAX = "max"
MIN = "min"
MEAN = "mean"
L2NORM = "l2norm"
STAT_CPM = [MAX, MIN, MEAN, L2NORM]

MAX_STAT_ABSOLUTE_ERROR = "max_stat_absolute_error"
MAX_STAT_RELATIVE_ERROR = "max_stat_relative_error"
MIN_STAT_ABSOLUTE_ERROR = "min_stat_absolute_error"
MIN_STAT_RELATIVE_ERROR = "min_stat_relative_error"
MEAN_STAT_ABSOLUTE_ERROR = "mean_stat_absolute_error"
MEAN_STAT_RELATIVE_ERROR = "mean_stat_relative_error"
L2NORM_STAT_ABSOLUTE_ERROR = "l2norm_stat_absolute_error"
L2NORM_STAT_RELATIVE_ERROR = "l2norm_stat_relative_error"
CSV_STATISTICS_HEADER = [
    TOKEN_ID,
    DATA_ID,
    GOLDEN_DATA_PATH,
    GOLDEN_OP_TYPE,
    MY_DATA_PATH,
    MY_OP_TYPE,
    MAX_STAT_ABSOLUTE_ERROR,
    MAX_STAT_RELATIVE_ERROR,
    MIN_STAT_ABSOLUTE_ERROR,
    MIN_STAT_RELATIVE_ERROR,
    MEAN_STAT_ABSOLUTE_ERROR,
    MEAN_STAT_RELATIVE_ERROR,
    L2NORM_STAT_ABSOLUTE_ERROR,
    L2NORM_STAT_RELATIVE_ERROR
]

TORCHAIR_FX_DATA_DIRECTORY = 'msit_fx_dump'
TORCHAIR_GE_DATA_DIRECTORY = 'msit_ge_dump'


def get_timestamp_sync():
    max_timestamp = int(datetime.datetime.now(tz=datetime.timezone.utc).strftime("%s"))
    str_world_size = os.environ.get("LOCAL_WORLD_SIZE", "1")
    world_size = safe_int(str_world_size, "LOCAL_WORLD_SIZE")
    if world_size < 2:
        return max_timestamp

    str_rank = os.environ.get("LOCAL_RANK", "0")
    rank = safe_int(str_rank, "LOCAL_RANK")
    max_timestamp = torch.tensor(max_timestamp)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=GLOBAL_DIST_BACKEND, rank=rank, world_size=world_size)
        torch.distributed.all_reduce(max_timestamp, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.destroy_process_group()
    else:
        torch.distributed.all_reduce(max_timestamp, op=torch.distributed.ReduceOp.MAX)

    return max_timestamp.item()


def get_ait_dump_path():
    global GLOBAL_AIT_DUMP_PATH

    if GLOBAL_AIT_DUMP_PATH == "msit_dump":
        max_timestamp = get_timestamp_sync()
        timestamp = datetime.datetime.fromtimestamp(max_timestamp).strftime("%Y%m%d_%H%M%S")
        os.environ[ATB_TIMESTAMP] = timestamp
        GLOBAL_AIT_DUMP_PATH = "msit_dump_" + timestamp

    return GLOBAL_AIT_DUMP_PATH
