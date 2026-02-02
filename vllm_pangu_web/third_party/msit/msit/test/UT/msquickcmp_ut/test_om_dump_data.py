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
import shutil
import stat
from unittest import mock

import pytest
import torch
from torch.onnx.utils import export
import numpy as np

from msquickcmp.npu.npu_dump_data import NpuDumpData, DynamicInput
from msquickcmp.npu.om_parser import OmParser
from msquickcmp.common.utils import AccuracyCompareException, parse_input_shape_to_list

FAKE_DYM_SHAPE_ONNX_MODEL_PATH = "fake_dym_shape_test_onnx_model.onnx"

FAKE_OM_MODEL_WITH_AIPP_PATH = "fake_with_aipp_test_onnx_model.om"

FAKE_OM_MODEL_PATH = "fake_test_onnx_model.om"
OM_OUT_PATH = FAKE_OM_MODEL_PATH.replace(".om", "")


FAKE_ONNX_MODEL_PATH = "fake_msquickcmp_test_onnx_model.onnx"
OUT_PATH = FAKE_ONNX_MODEL_PATH.replace(".onnx", "")
INPUT_SHAPE = (1, 3, 32, 32)

WRITE_FLAGS = os.O_WRONLY | os.O_CREAT  # 注意根据具体业务的需要设置文件读写方式
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR  # 注意根据具体业务的需要设置文件权限


class Args:
    def __init__(self, **kwargs):
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)


@pytest.fixture(scope="function")
def fake_arguments():
    return Args(
        model_path=FAKE_OM_MODEL_PATH,
        offline_model_path=FAKE_OM_MODEL_PATH,
        out_path=OM_OUT_PATH,
        cann_path="",
        input_shape="",
        input_path="",
        dump=True,
        output_size="",
        device="0",
        single_op=False,
    )


@pytest.fixture(scope="module", autouse=True)
def width_onnx_model():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 1, 1),
        torch.nn.BatchNorm2d(32),
        torch.nn.Conv2d(32, 32, 3, 2),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 32),
        torch.nn.Linear(32, 10),
    )
    export(model, torch.ones(INPUT_SHAPE), FAKE_ONNX_MODEL_PATH)
    yield FAKE_ONNX_MODEL_PATH

    if os.path.exists(FAKE_ONNX_MODEL_PATH):
        os.remove(FAKE_ONNX_MODEL_PATH)
    if os.path.exists(OUT_PATH):
        shutil.rmtree(OUT_PATH)


@pytest.fixture(scope="module", autouse=True)
def fake_dym_shape_onnx_model():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 1, 1),
        torch.nn.BatchNorm2d(32),
        torch.nn.Conv2d(32, 32, 3, 2),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 32),
        torch.nn.Linear(32, 10),
    )
    input_name = 'input0'
    input_data = torch.ones(INPUT_SHAPE)

    torch.onnx.export(
        model,
        input_data,
        FAKE_DYM_SHAPE_ONNX_MODEL_PATH,
        input_names=[input_name],
        dynamic_axes={input_name: {0: 'bs'}},
    )

    yield FAKE_DYM_SHAPE_ONNX_MODEL_PATH

    if os.path.exists(FAKE_DYM_SHAPE_ONNX_MODEL_PATH):
        os.remove(FAKE_DYM_SHAPE_ONNX_MODEL_PATH)


@pytest.fixture(scope="module", autouse=True)
def fake_om_model_with_aipp():
    with os.fdopen(os.open("fake_aipp.config", WRITE_FLAGS, WRITE_MODES), 'w') as file:
        if file is not None:
            aipp_lines = [
                "aipp_op{\n"
                "    aipp_mode:static\n"
                "    input_format : RGB888_U8\n"
                "    src_image_size_w : 32\n"
                "    src_image_size_h : 32\n"
                "    crop:false\n"
                "    min_chn_0 : 123.675\n"
                "    min_chn_1 : 116.28\n"
                "    min_chn_2 : 103.53\n"
                "    var_reci_chn_0 : 0.0171247538316637\n"
                "    var_reci_chn_1 : 0.0175070028011204\n"
                "    var_reci_chn_2 : 0.0174291938997821\n"
                "}"
            ]
            file.writelines(aipp_lines)
            file.close()

        yield FAKE_OM_MODEL_WITH_AIPP_PATH

        if os.path.exists(FAKE_OM_MODEL_WITH_AIPP_PATH):
            os.remove(FAKE_OM_MODEL_WITH_AIPP_PATH)

        if os.path.exists("fake_aipp.config"):
            os.remove("fake_aipp.config")


def test_init_given_valid_when_any_then_pass(fake_arguments):
    with mock.patch('msquickcmp.atc.atc_utils.convert_model_to_json'), \
         mock.patch.object(OmParser, '__init__', return_value=None), \
         mock.patch.object(DynamicInput, '__init__', return_value=None):
        aa = NpuDumpData(fake_arguments, False)

        assert aa.offline_model_path == "fake_test_onnx_model.om"
    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)


def test_init_given_invalid_when_any_then_failed(fake_arguments):
    fake_arguments.offline_model_path = ""

    with pytest.raises(AccuracyCompareException):
        aa = NpuDumpData(fake_arguments, False)

    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)


def test_generate_inputs_data_given_input_path_when_golden_then_pass(fake_arguments):
    tmp_input_data = "tmp_input_data"
    if not os.path.exists(tmp_input_data):
        os.makedirs(tmp_input_data, mode=0o700)

    input_path = os.path.join(tmp_input_data, "input_0.bin")
    input_data = np.random.uniform(size=INPUT_SHAPE).astype("float32")
    input_data.tofile(input_path)
    fake_arguments.input_path = input_path
    with mock.patch('msquickcmp.atc.atc_utils.convert_model_to_json'), \
         mock.patch.object(OmParser, '__init__', return_value=None), \
         mock.patch.object(DynamicInput, '__init__', return_value=None):
        npu_dump = NpuDumpData(fake_arguments, True)

        npu_dump.generate_inputs_data()

        assert os.path.exists(os.path.join(fake_arguments.out_path, "input"))

        inputs_list = parse_input_shape_to_list(fake_arguments.input_shape)
        input_bin_files = os.listdir(os.path.join(fake_arguments.out_path, "input"))

        for input_file, input_shape in zip(input_bin_files, inputs_list):
            input_data = np.fromfile(input_file)
            assert np.prod(input_data.shape) == np.prod(input_shape)

    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)
    if os.path.exists(tmp_input_data):
        shutil.rmtree(tmp_input_data)


def test_generate_inputs_data_given_random_data_when_aipp_then_pass(fake_arguments, fake_om_model_with_aipp):
    fake_arguments.offline_model_path = fake_om_model_with_aipp
    fake_arguments.out_path = fake_om_model_with_aipp.replace(".om", "")
    with mock.patch('msquickcmp.atc.atc_utils.convert_model_to_json'), \
         mock.patch.object(OmParser, '__init__', return_value=None), \
         mock.patch.object(DynamicInput, '__init__', return_value=None):
        npu_dump = NpuDumpData(fake_arguments, False)
        inputs_list = parse_input_shape_to_list(fake_arguments.input_shape)
        input_bin_files = os.listdir(os.path.join(fake_arguments.out_path, "input"))

        for input_file, input_shape in zip(input_bin_files, inputs_list):
            input_data = np.fromfile(input_file)
            assert np.prod(input_data.shape) == np.prod(input_shape)

    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)


def test_generate_dump_data_given_random_data_when_valid_then_pass(fake_arguments):
    with mock.patch('msquickcmp.atc.atc_utils.convert_model_to_json'), \
         mock.patch.object(OmParser, '__init__', return_value=None), \
         mock.patch.object(DynamicInput, '__init__', return_value=None), \
         mock.patch.object(NpuDumpData, 'benchmark_run', return_value=("./", "./")):
        npu_dump = NpuDumpData(fake_arguments, False)
        os.system(f"chmod -R 750 {OM_OUT_PATH}")
        om_dump_data_dir, _ = npu_dump.generate_dump_data()
        assert os.path.exists(om_dump_data_dir)

        assert len(os.listdir(om_dump_data_dir)) > 0

    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)
