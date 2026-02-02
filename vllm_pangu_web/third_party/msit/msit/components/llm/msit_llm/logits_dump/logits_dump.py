# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd. All rights reserved.
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

import pandas as pd

from msit_llm.common.log import logger
from components.utils.util import filter_cmd


HUMANEVAL_X_KEY_PREFIX = ["CPP", "Go", "Java", "JavaScript", "Python"]
HUMANEVAL_KEY_PREFIX = "HumanEval"


# 生成humaneval-X数据集的bad case list
def build_humanevalx_bad_case_list(key_list):
    bad_case_list = {key: [] for key in HUMANEVAL_X_KEY_PREFIX}

    for key in key_list:
        prefix = key.split('/')[0]
        if prefix not in HUMANEVAL_X_KEY_PREFIX:
            raise ValueError(f"Humaneval-X must has prefix in 'CPP/Go/Java/JavaScript/Python' (got {key})")
        try:
            bad_case_idx = int(key.split('/')[1])
        except ValueError as e:
            raise ValueError("The format of key column is wrong, please check bad_case_result_csv") from e
        bad_case_list[prefix].append(bad_case_idx)

    return list(bad_case_list.values())


# 生成humaneval数据集的bad case list
def build_humaneval_bad_case_list(key_list):
    bad_case_list = []
    for key in key_list:
        if key.split('/')[0] != HUMANEVAL_KEY_PREFIX:
            raise ValueError("Humaneval must has prefix 'HumanEval'")
        try:
            bad_case_idx = int(key.split('/')[1])
        except ValueError as e:
            raise ValueError("The format of key column is wrong, please check bad_case_result_csv") from e
        bad_case_list.append(bad_case_idx)
    return bad_case_list


# 生成BoolQ、Ceval、mmlu、gsm8k、needlebench数据集的bad case list
def build_others_bad_case_list(key_list):
    bad_case_list = []
    for key in key_list:
        if not isinstance(key, int) or key < 0:
            raise ValueError("The values in the key column must be of the int type"
                             " and greater than or equal to 0")
        bad_case_list.append(key)
    return bad_case_list


def build_bad_case_list(bad_case_csv_path):
    logger.info("Extract the key column to build BAD_CASE_LIST from bad case csv result")
    if not isinstance(bad_case_csv_path, str) or not bad_case_csv_path.endswith('.csv'):
        raise ValueError("Invalid csv path, only path with suffix '.csv' is allowed: %r" % bad_case_csv_path)
    bad_case_df = pd.read_csv(bad_case_csv_path)
    key_list = bad_case_df['key'].tolist()
    if len(key_list) == 0:
        raise RuntimeError("Bad case list is empty, no need to dump logits")

    bad_case_list = []
    if isinstance(key_list[0], str) and '/' in key_list[0]:
        if key_list[0].split('/')[0] in HUMANEVAL_X_KEY_PREFIX:
            logger.info("Build humaneval-X dataset bad case list")
            bad_case_list = build_humanevalx_bad_case_list(key_list)
        elif key_list[0].split('/')[0] == HUMANEVAL_KEY_PREFIX:
            logger.info("Build humaneval dataset bad case list")
            bad_case_list = build_humaneval_bad_case_list(key_list)
        else:
            raise RuntimeError("When the 'key' column is of character type, "
                               "only the HumanEval and HumanEvalX datasets are supported.")
    elif isinstance(key_list[0], int):
        logger.info("Build BoolQ/mmlu/gsm8k/needlebench dataset bad case list")
        bad_case_list = build_others_bad_case_list(key_list)
    else:
        raise TypeError("The values in the key column must be strings with the prefix for"
                        " 'humaneval' or 'humanevalx', or all be integers.")
    logger.info("Build bad case list success")
    return bad_case_list


def check_gpu():
    try:
        import torch
    except Exception:
        return False
    if torch.cuda.is_available():
        return True
    else:
        return False


def check_npu():
    try:
        import torch_npu
    except Exception:
        return False
    if torch_npu.npu.is_available():
        return True
    else:
        return False


def execute_command(cmd, info_need=True):
    try:
        cmd = filter_cmd(cmd)
        if info_need:
            logger.info("Execute command: " + " ".join(cmd))
        subprocess.run(cmd, shell=False, check=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        logger.error(f"Failed to execute modeltest cmd, error code: {e.returncode}")
        logger.error("Error message in modeltest: ")
        logger.error(e.stdout)
        raise RuntimeError("Failed to execute modeltest cmd") from e


def del_env():
    del os.environ['BAD_CASE_LOGITS_DUMP']
    del os.environ['LOGITS_DUMP_TOKEN_MAX_LENGTH']
    del os.environ['BAD_CASE_LIST']


class LogitsDumper:
    def __init__(self, args):
        self.exec = args.exec
        self.bad_case_result_csv = args.bad_case_result_csv
        self.token_range = args.token_range

    def dump_logits(self):
        bad_case_list = build_bad_case_list(self.bad_case_result_csv)
        os.environ['BAD_CASE_LOGITS_DUMP'] = "True"
        os.environ['LOGITS_DUMP_TOKEN_MAX_LENGTH'] = f"{self.token_range}"
        os.environ['BAD_CASE_LIST'] = repr(bad_case_list)
        cmd_model_infer = self.exec.split()
        if len(cmd_model_infer) < 1:
            raise RuntimeError("The input command is empty.")
        elif cmd_model_infer[0] != "torchrun" and cmd_model_infer[0] != "modeltest":
            raise RuntimeError("The input command is invaild."
                               " It needs to start with either 'torchrun' or 'modeltest'.")
        if not (check_npu() or check_gpu()):
            del_env()
            raise RuntimeError("NPU/GPU is not available")

        execute_command(cmd_model_infer, True)

        logger.info("'Logits Dump' has successfully finished, the logits is stored at '%s'",
                    "{{output_dir}}/data/{{DEVICE_TYPE}}/precision_test/{{dataset}}/"
                    "{{data_type}}/{{model_name}}/logits/")
        logger.info("The logits dump process is finished. Eliminate the impact of the environment variables.")
        del_env()
