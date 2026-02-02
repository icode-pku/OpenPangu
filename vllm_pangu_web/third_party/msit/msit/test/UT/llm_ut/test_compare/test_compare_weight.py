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
from glob import glob
from unittest import mock

import pytest
import pandas as pd
import torch
from safetensors.torch import save_file  

torch.manual_seed(0)


FILE_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
FAKE_GOLDEN_DATA_PATH = "./test_acc_cmp_fake_golden_data"
FAKE_MY_DATA_PATH = "./test_acc_cmp_fake_test_data"
OUTPUT_CSV_PATH = "./output_csv_path"


@pytest.fixture(scope='module', autouse=True)
def test_fake_golden_data_path():

    if not os.path.exists(FAKE_GOLDEN_DATA_PATH):
        os.makedirs(FAKE_GOLDEN_DATA_PATH, mode=0o750)
        tensors = {"weight":torch.randint(0, 1, (2, 2), dtype=torch.int8).to(dtype=torch.float32)}
        save_file(tensors, os.path.join(FAKE_GOLDEN_DATA_PATH, "model.safetensors"))
    yield FAKE_GOLDEN_DATA_PATH

    if os.path.exists(FAKE_GOLDEN_DATA_PATH):
        shutil.rmtree(FAKE_GOLDEN_DATA_PATH)


@pytest.fixture(scope='module', autouse=True)
def test_fake_quant_data_path():

    if not os.path.exists(FAKE_MY_DATA_PATH):
        os.makedirs(FAKE_MY_DATA_PATH, mode=0o750)
        tensors = {"weight":torch.randint(0, 1, (2, 2), dtype=torch.int8),
                   "weight_offset":torch.zeros(1),
                   "weight_scale":torch.ones(1)
                   }
        save_file(tensors, os.path.join(FAKE_MY_DATA_PATH, "model.safetensors"))
    yield FAKE_MY_DATA_PATH

    if os.path.exists(FAKE_MY_DATA_PATH):
        shutil.rmtree(FAKE_MY_DATA_PATH)


def test_compare_weight_given_loaded_data_when_valid_then_pass():
    from msit_llm.compare.cmp_weight import compare_weight
    if not os.path.exists(OUTPUT_CSV_PATH):
        os.makedirs(OUTPUT_CSV_PATH, mode=0o750)
    compare_weight(FAKE_GOLDEN_DATA_PATH, FAKE_MY_DATA_PATH, OUTPUT_CSV_PATH)
    csv_file_path = glob(f'{OUTPUT_CSV_PATH}/*.csv')[0]
    row_data = pd.read_csv(csv_file_path)
    assert row_data["cosine_similarity"][0] == 1.0

    if os.path.exists(OUTPUT_CSV_PATH):
        shutil.rmtree(OUTPUT_CSV_PATH)