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


import os
import argparse
import re
from collections import namedtuple

from components.utils.constants import PATH_WHITE_LIST_REGEX, FileCheckConst
from components.utils.util import check_file_size_based_on_ext, check_file_ext, safe_int
from components.utils.file_open_check import FileStat
from components.utils.file_utils import FileChecker
from msit_llm.common.constant import MAX_DATA_SIZE
from components.utils.check.rule import Rule
from msit_llm.common.log import logger

STR_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{: /.~-]")
INVALID_CHARS = ['|', ';', '&', '&&', '||', '>', '>>', '<', '`', '\\', '!', '\n']

NAMEDTUPLE_PRECISION_METRIC = namedtuple('precision_metric', ['abs', 'kl', 'cos_sim'])('abs', 'kl', 'cos_sim')
NAMEDTUPLE_PRECISION_MODE = namedtuple(
    'precision_mode', ["keep_origin_dtype", "force_fp16", "force_fp32"]
)("keep_origin_dtype", "force_fp16", "force_fp32")

DEVICE_COUNT_MAX = 256
DEVICE_NUM_MIN = 0
DEVICE_NUM_MAX = 255


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected true, 1, false, 0 with case insensitive.')


def check_positive_integer(value):
    ivalue = safe_int(value)
    if ivalue < 0 or ivalue > 2:
        raise argparse.ArgumentTypeError("%s is an invalid int value" % value)
    return ivalue


def check_dump_time_integer(value):
    ivalue = safe_int(value)
    if ivalue < 0 or ivalue > 3:
        raise argparse.ArgumentTypeError("%s is an invalid int value" % value)
    return ivalue


def check_device_integer(value):
    ivalue = safe_int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid int value" % value)
    return ivalue


def check_process_integer(value):
    ivalue = safe_int(value)
    if ivalue < 1 or ivalue > 8:
        raise argparse.ArgumentTypeError("%s is an invalid int value. The number of processes should be 1~8." % value)
    return ivalue


def safe_string(value):
    if not value:
        return value
    if re.search(STR_WHITE_LIST_REGEX, value):
        raise ValueError("String parameter contains invalid characters.")
    return value


def check_number_list(value):
    # just like "1241414,124141,124424"
    if not value:
        return value
    outsize_list = value.split(',')
    for outsize in outsize_list:
        regex = re.compile(r"[^0-9]")
        if regex.search(outsize):
            raise argparse.ArgumentTypeError(f"output size \"{outsize}\" is not a legal number")
    return value


def check_ids_string(value):
    if not value:
        return value
    dym_string = value
    pattern = r'^(\d+(?:_\d+)*)(,\d+(?:_\d+)*)*$'
    if not re.match(pattern, dym_string):
        raise argparse.ArgumentTypeError(f"dym range string \"{dym_string}\" is not a legal string")
    return dym_string


def check_exec_script_file(script_path: str):
    if not os.path.exists(script_path):
        raise argparse.ArgumentTypeError(f"Script Path does not exist : {script_path}")


def check_input_args(args: list):
    for arg in args:
        for char in arg:
            if char in INVALID_CHARS:
                raise argparse.ArgumentTypeError("Args has invalid chars. Please check")


def check_exec_cmd(command: str):
    cmds = command.split()
    if len(cmds) < 2:
        raise argparse.ArgumentTypeError(f"Run cmd is not valid: \"{command}\" ")
    elif len(cmds) == 2:
        script_file = cmds[1]
        check_exec_script_file(script_file)
    else:
        script_file = cmds[1]
        check_exec_script_file(script_file)
        args = cmds[2:]
        check_input_args(args)
    return True


def check_output_path_legality(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError("output path is illegal. Please check.") from err
    if not file_stat.is_basically_legal("write", strict_permission=False):
        raise argparse.ArgumentTypeError("output path can not be written. Please check.")
    return path_value


def check_input_path_legality(value):
    if not value:
        return value
    inputs_list = value.split(',')
    for input_path in inputs_list:
        try:
            file_stat = FileStat(input_path)
        except Exception as err:
            raise argparse.ArgumentTypeError(f"input path:{input_path} is illegal. Please check.") from err
        if not file_stat.is_basically_legal('read', strict_permission=False):
            raise argparse.ArgumentTypeError(f"input path:{input_path} is illegal. Please check.")
    return value


def check_config_path_legality(config_path):
    file_check = FileChecker(config_path, FileCheckConst.FILE, ability=FileCheckConst.READ_ABLE)
    file_check.common_check()
    return config_path


def check_data_file_size(data_path, max_size=MAX_DATA_SIZE):
    try:
        file_stat = FileStat(data_path)
    except Exception as e:
        raise Exception(f"data path: {data_path} is illegal. Please check it.") from e

    if not file_stat.is_legal_file_size(max_size):
        raise Exception(f"the size of file: {data_path} is out of max limit {max_size} byte.")

    return True


def check_data_can_convert_to_int(value):
    try:
        result = Rule.to_int().check(value)
    except Exception as err:
        raise argparse.ArgumentTypeError("%s can not convert to int." % value) from err
    if not result:
        raise argparse.ArgumentTypeError("%s can not convert to int." % value)
    return int(value)


def load_file_to_read_common_check(path: str, exts=None):
    if not isinstance(path, str):
        raise TypeError("'path' should be 'str'")

    if isinstance(exts, (tuple, list)):
        if not any(check_file_ext(path, ext) for ext in exts):
            logger.error("Expected extenstion to be one of %r", exts)
            raise ValueError

    elif exts is not None:
        logger.error("Expected 'exts' to be 'List[str]', got %r instead", type(exts))
        raise TypeError

    if re.search(PATH_WHITE_LIST_REGEX, path):
        logger.error("Invalid character: %r", path)
        raise ValueError

    path = os.path.realpath(path)

    try:
        file_status = os.stat(path)
    except OSError as e:
        logger.error("%s: %r", e.strerror, path)
        raise

    if not os.st.S_ISREG(file_status.st_mode):
        logger.error("Not a regular file: %r", path)
        raise ValueError

    if not check_file_size_based_on_ext(path):
        logger.error("File too large: %r", path)
        raise ValueError

    if (os.st.S_IWOTH & file_status.st_mode) == os.st.S_IWOTH:
        logger.error("Vulnerable path: %r should not be other writeable", path)
        raise PermissionError

    safe_parent_msg = Rule.path().is_safe_parent_dir().check(path)
    if not safe_parent_msg:
        raise PermissionError(f"parent dir of {path} is not safe. {str(safe_parent_msg)}")

    cur_euid = os.geteuid()
    if file_status.st_uid != cur_euid:
        # not root
        if cur_euid != 0:
            logger.error("File owner and current user are inconsistent: %r", path)
            raise PermissionError

        # root but reading a other writeable file
        elif (os.st.S_IWGRP & file_status.st_mode) == os.st.S_IWGRP or \
             (os.st.S_IWUSR & file_status.st_mode) == os.st.S_IWUSR:
            logger.warning("Privilege escalation risk detected. Trying to read a file that belongs to"
                           " a normal user and is writeable to the user or the user group")

    return path


def check_device_integer_range_valid(device_id):
    if device_id < DEVICE_NUM_MIN or device_id > DEVICE_NUM_MAX:
        raise argparse.ArgumentTypeError("device id: {} is invalid. valid value range is [{}, {}]".format(
                        device_id, DEVICE_NUM_MIN, DEVICE_NUM_MAX))


def check_device_range_valid(value):
    # if contain , split to int list
    try:
        # Check if the value contains a comma; if so, split into a list of integers
        if ',' in value:
            ilist = [int(v) for v in value.split(',')]
            if len(ilist) > DEVICE_COUNT_MAX:
                raise argparse.ArgumentTypeError(
                    f"too much device id in --device, max permitted count is {DEVICE_COUNT_MAX}"
                )
            for ivalue in ilist:
                check_device_integer_range_valid(ivalue)
            return ilist
        else:
            # default as single int value
            if not value.isdigit():
                raise argparse.ArgumentTypeError("%s contains special characters other than numbers." % value)
            ivalue = int(value)
            check_device_integer_range_valid(ivalue)
            return ivalue
    except ValueError as e:
        raise argparse.ArgumentTypeError("Argument npu-id invalid input value: {}. "
                                         "Please provide a valid integer or a comma-separated list of integers."
                                         .format(value)) from e


def check_token_range(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError("Token range must greater than 0")
    return ivalue


def check_cosine_similarity(value):
    try:
        ivalue = float(value)
    except Exception as err:
        raise argparse.ArgumentTypeError("Cosine similarity is invalid. Please check.") from err
    if ivalue < -1 or ivalue > 1:
        raise argparse.ArgumentTypeError("Cosine similarity: %s is an invalid float value" % value)
    return ivalue


def check_kl_divergence(value):
    try:
        ivalue = float(value)
    except Exception as err:
        raise argparse.ArgumentTypeError("KL_divergence is invalid. Please check.") from err
    if ivalue < 0:
        raise argparse.ArgumentTypeError("KL_divergence: %s is an invalid float value" % value)
    return ivalue


def check_l1_norm(value):
    try:
        ivalue = float(value)
    except Exception as err:
        raise argparse.ArgumentTypeError("L1_Norm is invalid. Please check.") from err
    if ivalue < -1:
        raise argparse.ArgumentTypeError("L1_Norm: %s is an invalid float value" % value)
    return ivalue


def safe_int_env(var_name, default):
    """Safely convert an environment variable to an integer with error handling."""
    value = os.getenv(var_name, str(default)).strip()  # 获取环境变量并去除前后空格
    try:
        return int(value)
    except (ValueError, TypeError) as e:
        logger.warning("Failed to convert '%r' for %r, using default %r. Error: %r" % (value, var_name, default, e))
        return default


def check_specified_rank_id(value):
    if value is None:
        return value
    if not str.isdigit(value):
        raise argparse.ArgumentTypeError(f"rank id:{value} is illegal. Please check.")
    return int(value)


def get_rank_id_from_torchair_data(dir_name: str):
    rank_id = -1
    rank_index = dir_name.rfind('rank')
    if dir_name.startswith('worldsize') and rank_index != -1 and str.isdigit(dir_name[rank_index + 4:]):
        rank_id = int(dir_name[rank_index + 4:])

    return rank_id
