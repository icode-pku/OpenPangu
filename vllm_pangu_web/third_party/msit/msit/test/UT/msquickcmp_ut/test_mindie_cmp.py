# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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
import stat
from unittest.mock import patch
import pandas as pd
import pytest
import torch
from components.debug.compare.msquickcmp.mie_torch.mietorch_comp import MIETorchCompare
from components.debug.compare.utils.ge_dump_reader import GEDumpFileReader
from components.debug.compare.utils.torch_dump_reader import TorchDumpFileReader


FAKE_GOLDEN_DATA_PATH = "test_resource"
FAKE_MY_DATA_PATH = "test_resource"
FAKE_RT_OP_MAPPING_JSON_PATH = "mindie_rt_op_mapping.json"
FAKE_TORCH_OP_MAPPING_JSON_PATH = "mindie_torch_op_mapping.json"


flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL  
mode = stat.S_IWUSR | stat.S_IRUSR  


@pytest.fixture(scope="function")
def create_golden_data_and_my_data():
    my_data = torch.randn(3, 3, dtype=torch.float32)
    golden_data = torch.randn(3, 3, dtype=torch.float32)
    bad_my_data = torch.randn(2, 3, dtype=torch.float32)
    nan_golden_data = torch.full((3, 3), float('nan'))
    golden_data_and_my_data = [my_data, golden_data, bad_my_data, nan_golden_data]
    return golden_data_and_my_data


@pytest.fixture(scope='module')
def rt_op_mapping_file():
    rt_op_mapping_data = [{
        "fusion_op": "Conv2D_5Relu_8",
        "fusion_ops": [
            "Conv2D_5",
            "Relu_8"
        ],
        "ge_op": "Conv2D_5",
        "id": 4,
        "rt_layer": "CONVOLUTION_5"
    },
    {
        "fusion_op": "Conv2D_5Relu_8",
        "fusion_ops": [
            "Conv2D_5",
            "Relu_8"
        ],
        "ge_op": "Conv2D_5",
        "id": 5,
        "rt_layer": "ACTIVATION_6"
    }]
    current_file_path = os.path.dirname(__file__)
    rt_op_mapping_json_path = os.path.join(current_file_path, FAKE_RT_OP_MAPPING_JSON_PATH)
    with os.fdopen(os.open(rt_op_mapping_json_path, flags, mode), 'w') as f:
        json.dump(rt_op_mapping_data, f, indent=4)

    yield rt_op_mapping_json_path
    
    if os.path.exists(rt_op_mapping_json_path):
        os.remove(rt_op_mapping_json_path)


@pytest.fixture(scope='module')
def torch_op_mapping_file():
    torch_op_mapping_data = [{
        "jit_node": "%input.9 : Float(1, 64, 56, 56) = aten::_convolution(%input.7, \
        %self.layer1.0.conv1.weight.1_fused_bn, %9_fused_bn.3, %6, %5, %6, %11, %5, \
        %10, %11, %11, %12, %12), scope: __module.layer1/__module.layer1.0/__module.layer1.0.conv1 \
        # /usr/local/python3.10.2/lib/python3.10/site-packages/torch/nn/modules/conv.py:456:0",
        "rt_layer": "CONVOLUTION_5"
    },
    {
        "jit_node": "%968 : Float(1, 64, 56, 56) = aten::relu(%input.9), scope: \
        __module.layer1/__module.layer1.0/__module.layer1.0.relu \
        # /usr/local/python3.10.2/lib/python3.10/site-packages/torch/nn/functional.py:1469:0",
        "rt_layer": "ACTIVATION_6"
    }]
    current_file_path = os.path.dirname(__file__)
    torch_op_mapping_json_path = os.path.join(current_file_path, FAKE_TORCH_OP_MAPPING_JSON_PATH)
    with os.fdopen(os.open(torch_op_mapping_json_path, flags, mode), 'w') as f:
        json.dump(torch_op_mapping_data, f, indent=4)

    yield torch_op_mapping_json_path

    if os.path.exists(torch_op_mapping_json_path):
        os.remove(torch_op_mapping_json_path)
    

def test_mindie_torch_compare(rt_op_mapping_file, torch_op_mapping_file):
    current_file_path = os.path.dirname(__file__)
    copmarer = MIETorchCompare(FAKE_GOLDEN_DATA_PATH, FAKE_MY_DATA_PATH, current_file_path, current_file_path)

    test_data = torch.ones((1, 64, 56, 56), dtype=torch.float32)
    test_data[0, 0, 0, 0] = 10
    with patch.object(TorchDumpFileReader, 'get_tensor', return_value=test_data), \
         patch.object(GEDumpFileReader, 'get_tensor', return_value=test_data.numpy()):
        copmarer.compare()
        csv_file_path = os.path.join(current_file_path, "comparison_results.csv")
        assert os.path.isfile(csv_file_path)
        df = pd.read_csv(csv_file_path)
        assert df.loc[0, "cosine_similarity"] == 1.0
    
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
    new_json_file = os.path.join(current_file_path, "op_map_updated.json")
    if os.path.exists(new_json_file):
        os.remove(new_json_file)


def test_check_tensor(create_golden_data_and_my_data):
    my_data, golden_data, bad_golden_data, nan_golden_data = create_golden_data_and_my_data
    cpu_tensor = golden_data.reshape(-1)
    npu_tensor = my_data.reshape(-1)
    nan_cpu_tensor = nan_golden_data.reshape(-1)
    nan_npu_tensor = nan_golden_data.reshape(-1)
    bad_npu_tensor = bad_golden_data.reshape(-1)
    bad_nan_cpu_tensor = torch.full((2, 2), float('nan')).reshape(-1)

    # test tensor shape
    tensor_pass1, message1 = MIETorchCompare.check_tensor(cpu_tensor, npu_tensor)
    tensor_pass2, message2 = MIETorchCompare.check_tensor(cpu_tensor, bad_npu_tensor)
    # test isfinite check
    tensor_pass3, message3 = MIETorchCompare.check_tensor(nan_cpu_tensor, npu_tensor)
    tensor_pass4, message4 = MIETorchCompare.check_tensor(cpu_tensor, nan_npu_tensor)
    tensor_pass5, message5 = MIETorchCompare.check_tensor(bad_nan_cpu_tensor, nan_npu_tensor)

    assert (tensor_pass1, message1) == (True, "")
    assert (tensor_pass2, message2) == (False, "data shape doesn't match.")
    assert (tensor_pass3, message3) == (False, "cpu_data includes NAN or inf.")
    assert (tensor_pass4, message4) == (False, "npu_data includes NAN or inf.")
    assert (tensor_pass5, message5) == (
        False, "data shape doesn't match. cpu_data includes NAN or inf. npu_data includes NAN or inf."
    )