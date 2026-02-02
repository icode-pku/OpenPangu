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
import json
import csv
from pathlib import Path
import logging
from collections import namedtuple
from glob import glob
from msserviceprofiler.msguard.security.io import open_s
from msserviceprofiler.msguard import validate_params, Rule

TARGETS = namedtuple("TARGETS", ["FirstTokenTime", "Throughput"])("FirstTokenTime", "Throughput")
_SUGGESTION_TYPES = ["env", "system", "config"]
SUGGESTION_TYPES = namedtuple("SUGGESTION_TYPES", _SUGGESTION_TYPES)(*_SUGGESTION_TYPES)
MAX_FILE_ITER_TIME = 10000
MAX_FILE_SIZE = 10
BYTES_TO_GB = 1024**3
MAX_DEVICE_ID_LIST_LENGTH = 128

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL,
}


def str_ignore_case(value):
    return value.lower().replace("_", "").replace("-", "")


def get_dict_value_by_pos(dict_value, target_pos):
    cur = dict_value
    for kk in target_pos.split(":"):
        if not cur:
            cur = None
            break
        if isinstance(cur, list) and str.isdigit(kk):
            cur = cur[int(kk)]
        elif kk in cur:
            cur = cur[kk]
        else:
            cur = None
            break
    return cur


def set_log_level(level="info"):
    if level.lower() in LOG_LEVELS:
        logger.setLevel(LOG_LEVELS.get(level.lower()))
    else:
        logger.warning("Set %r log level failed.", level)


def set_logger(msit_logger):
    msit_logger.propagate = False
    msit_logger.setLevel(logging.INFO)
    if not msit_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(formatter)
        msit_logger.addHandler(stream_handler)


logger = logging.getLogger("msservice_advisor_logger")
set_logger(logger)


def get_directory_size(path):
    iter_time = 0
    total_size = 0
    for dirpath, _, filename in os.walk(path):
        for f in filename:
            if iter_time > MAX_FILE_ITER_TIME:
                raise ValueError(f"path '{path}' iter times cannot exceed {MAX_FILE_ITER_TIME}.")
            iter_time += 1

            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size / BYTES_TO_GB


def get_latest_matching_file(instance_path, pattern):
    files = glob(os.path.join(instance_path, pattern))
    return max(files, key=os.path.getmtime) if files else None


def read_csv(file_path):
    logger.info("Reading CSV file: %r", file_path)
    result = {}
    try:
        with open_s(file_path, mode="r", newline="", encoding="utf-8") as ff:
            reader = csv.DictReader(ff)
            if not reader.fieldnames:
                logger.error("CSV file %r has no headers or is empty.", file_path)
                return None
            for row in reader:
                for kk, vv in row.items():
                    result.setdefault(kk, []).append(vv)
            if not result:
                logger.error("CSV file %r is empty or has no valid data.", file_path)
                return None
    except csv.Error as e:
        logger.error("CSV file %r is not properly formatted: %s", file_path, e)
        return None
    except Exception as e:
        logger.error("Failed to read CSV file %r: %s", file_path, e)
        return None
    return result


def read_json(file_path):
    logger.info("Reading JSON file: %r", file_path)
    try:
        with open_s(file_path, mode="r", encoding="utf-8") as ff:
            result = json.load(ff)
            if not isinstance(result, dict):
                logger.error("JSON file %r does not contain a JSON object.", file_path)
                return None
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON file %r: %s", file_path, e)
        return None
    except Exception as e:
        logger.error("Failed to read JSON file %r: %s", file_path, e)
        return None
    return result


@validate_params({'file_path': Rule.input_file_read})
def read_csv_or_json(file_path):
    if not file_path or not os.path.exists(file_path):
        logger.warning("File does not exist: %r", file_path)
        return None
    if file_path.endswith(".json"):
        return read_json(file_path)
    elif file_path.endswith(".csv"):
        return read_csv(file_path)
    else:
        logger.warning("Unsupported file format: %r", file_path)
        return None


class UmaskWrapper:
    """Write with preset umask
    >>> with UmaskWrapper():
    >>>     ...
    """

    def __init__(self, umask=0o027):
        self.umask, self.ori_umask = umask, None

    def __enter__(self):
        self.ori_umask = os.umask(self.umask)

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        os.umask(self.ori_umask)


def is_root():
    return os.name != 'nt' and os.getuid() == 0