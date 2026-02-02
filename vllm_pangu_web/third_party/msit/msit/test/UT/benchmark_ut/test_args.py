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
import stat
import argparse
import unittest
from unittest.mock import patch, MagicMock

import pytest
from msit_benchmark.__main__ import get_cmd_instance


FILE_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP

PREFIX = "benchmark_test_args_fake_"
FAKE_OM_PATH = PREFIX + "model.om"
FAKE_INVALID_OM_PATH = PREFIX + "invalid_model.om"
FAKE_NOT_EXISTS_OM_PATH = PREFIX + "not_exists_model.om"
FAKE_BIN_PATH = PREFIX + "data.bin"
FAKE_ACL_JSON_PATH = PREFIX + "acl.json"
FAKE_NOT_EXISTS_ACL_JSON_PATH = PREFIX + "not_exists_acl.json"
FAKE_AIPP_CFG_PATH = PREFIX + "aipp.config"

VALID_MODE = int("640", 8)
INVALID_MODE = int("770", 8)
INVALID_ARG = "--invalid_arg"


@pytest.fixture(scope="module", autouse=True)
def init_resources():
    file_names = [FAKE_OM_PATH, FAKE_INVALID_OM_PATH, FAKE_BIN_PATH, FAKE_ACL_JSON_PATH, FAKE_AIPP_CFG_PATH]
    for file_name in file_names:
        with os.fdopen(os.open(file_name, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), "w") as ff:
            pass
        mode = INVALID_MODE if file_name == FAKE_INVALID_OM_PATH else VALID_MODE
        os.chmod(file_name, mode)

    yield

    for file_name in file_names:
        if os.path.exists(file_name):
            os.chmod(file_name, INVALID_MODE)
            os.remove(file_name)


FULL_CMD_DICT = {
    "--om-model": FAKE_OM_PATH,
    "--input": FAKE_BIN_PATH,
    "--output": "output/",
    "--output-dirname": "outdir/",
    "--outfmt": "NPY",
    "--loop": "100",
    "--debug": "0",
    "--device": "0,1",
    "--dym-batch": "16",
    "--dym-hw": "224,224",
    "--dym-dims": "1,3,224,224",
    "--dym-shape": "1,3,224,224",
    "--output-size": "10000",
    "--auto-set-dymshape-mode": "0",
    "--auto-set-dymdims-mode": "0",
    "--batch-size": "16",
    "--pure-data-type": "zero",
    "--profiler": "0",
    "--dump": "0",
    "--acl-json-path": FAKE_ACL_JSON_PATH,
    "--output-batchsize-axis": "1",
    "--run-mode": "array",
    "--display-all-summary": "0",
    "--warmup-count": "1",
    "--dym-shape-range": "data:1~3,3,224,224-226",
    "--aipp-config": FAKE_AIPP_CFG_PATH,
    "--energy-consumption": "0",
    "--npu-id": "0",
    "--backend": "trtexec",
    "--perf": "0",
    "--pipeline": "0",
    "--profiler-rename": "0",
    "--dump-npy": "0",
    "--divide-input": "0",
    "--threads": "1",
}

SHORT_CMD_DICT = {
    "-om": FAKE_OM_PATH,
    "-i": FAKE_BIN_PATH,
    "-o": "output/",
    "-od": "outdir/",
    "--outfmt": "NPY",
    "--loop": "100",
    "--debug": "0",
    "-d": "0,1",
    "-db": "16",
    "-dhw": "224,224",
    "-dd": "1,3,224,224",
    "-ds": "1,3,224,224",
    "-outsize": "10000",
    "-asdsm": "0",
    "-asddm": "0",
    "--batch-size": "16",
    "-pdt": "zero",
    "-pf": "0",
    "--dump": "0",
    "-acl": FAKE_ACL_JSON_PATH,
    "-oba": "1",
    "-rm": "array",
    "-das": "0",
    "-wcount": "1",
    "-dr": "data:1~3,3,224,224-226",
    "-aipp": FAKE_AIPP_CFG_PATH,
    "-ec": "0",
    "--npu-id": "0",
    "--backend": "trtexec",
    "--perf": "0",
    "--pipeline": "0",
    "--profiler-rename": "0",
    "--dump-npy": "0",
    "--divide-input": "0",
    "--threads": "1",
}


def benchmark_argparse(argv):
    aa = get_cmd_instance()
    parser = argparse.ArgumentParser()
    aa.add_arguments(parser)
    args_parser = parser.parse_args(argv)
    return args_parser


def call_benchmark_cmd(argv):
    aa = get_cmd_instance()
    args = benchmark_argparse(argv)
    return aa.handle(args)


def cmd_dict_to_list(cmd_dict, new_args={}):
    cmd_list = []
    for key, value in cmd_dict.items():
        cmd_list.append(key)
        cmd_list.append(new_args.pop(key, value))

    for key, value in new_args.items():
        cmd_list.append(key)
        cmd_list.append(value)
    return cmd_list


def test_benchmark_argparse_given_valid_when_full_then_pass():
    args = benchmark_argparse(cmd_dict_to_list(FULL_CMD_DICT))
    assert args.om_model == FAKE_OM_PATH
    assert args.input == FAKE_BIN_PATH


def test_benchmark_argparse_given_valid_when_short_then_pass():
    args = benchmark_argparse(cmd_dict_to_list(SHORT_CMD_DICT))
    assert args.om_model == FAKE_OM_PATH
    assert args.input == FAKE_BIN_PATH


def test_benchmark_argparse_given_invalid_arg_when_full_then_error():
    new_args = {INVALID_ARG: "anything"}
    with pytest.raises(SystemExit) as e:
        args = benchmark_argparse(cmd_dict_to_list(FULL_CMD_DICT, new_args=new_args))


def test_benchmark_argparse_given_invalid_model_when_full_then_error():
    new_args = {"--om-model": FAKE_INVALID_OM_PATH}
    with pytest.raises(SystemExit) as e:
        args = benchmark_argparse(cmd_dict_to_list(FULL_CMD_DICT, new_args=new_args))


def test_benchmark_argparse_given_negative_loop_when_full_then_error():
    new_args = {"--loop": "-3"}
    with pytest.raises(SystemExit) as e:
        args = benchmark_argparse(cmd_dict_to_list(FULL_CMD_DICT, new_args=new_args))


def test_benchmark_argparse_given_negative_batch_size_when_full_then_error():
    new_args = {"--batch-size": "-3"}
    with pytest.raises(SystemExit) as e:
        args = benchmark_argparse(cmd_dict_to_list(FULL_CMD_DICT, new_args=new_args))


def test_benchmark_argparse_given_negative_warmup_count_when_full_then_error():
    new_args = {"--warmup-count": "-3"}
    with pytest.raises(SystemExit) as e:
        args = benchmark_argparse(cmd_dict_to_list(FULL_CMD_DICT, new_args=new_args))


def test_benchmark_argparse_given_negative_output_batchsize_axis_when_full_then_error():
    new_args = {"--output-batchsize-axis": "-3"}
    with pytest.raises(SystemExit) as e:
        args = benchmark_argparse(cmd_dict_to_list(FULL_CMD_DICT, new_args=new_args))


def test_benchmark_argparse_given_large_device_when_full_then_error():
    new_args = {"--device": "1,234,257"}
    with pytest.raises(SystemExit) as e:
        args = benchmark_argparse(cmd_dict_to_list(FULL_CMD_DICT, new_args=new_args))


def test_benchmark_argparse_given_invalid_outfmt_when_full_then_error():
    new_args = {"--outfmt": "JSON"}
    with pytest.raises(SystemExit) as e:
        args = benchmark_argparse(cmd_dict_to_list(FULL_CMD_DICT, new_args=new_args))
