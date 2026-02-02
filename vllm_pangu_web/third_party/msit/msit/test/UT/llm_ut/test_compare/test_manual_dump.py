# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

import os, json
from unittest import mock
from unittest.mock import patch
import pytest

import torch

from components.llm.msit_llm.dump.manual_dump import dump_data, write_json_file


def test_dump_data_with_invalid_params():
    # Test invalid token_id
    with mock.patch('msit_llm.common.log.logger.warning') as mocked_warning:
        dump_data(token_id=-1, data_id=1, golden_data=torch.tensor([1]), my_path="some/path", output_path="./")
        mocked_warning.assert_called_once_with('Please check whether token_id passed in are correct')
    # Test invalid data_id
    with mock.patch('msit_llm.common.log.logger.warning') as mocked_warning:
        dump_data(token_id=1, data_id=-1, golden_data=torch.tensor([1]), my_path="some/path", output_path="./")
        mocked_warning.assert_called_once_with('Please check whether data_id passed in are correct')
    # Test non-tensor golden_data
    with mock.patch('msit_llm.common.log.logger.warning') as mocked_warning:
        dump_data(token_id=1, data_id=1, golden_data=None, my_path="some/path", output_path="./")
        mocked_warning.assert_called_once_with('The golden_data is not a torch tensor!')
    # Test empty my_path
    with mock.patch('msit_llm.common.log.logger.warning') as mocked_warning:
        dump_data(token_id=1, data_id=1, golden_data=torch.tensor([1]), my_path="", output_path="./")
        mocked_warning.assert_called_once_with('Please check whether my_path passed in are correct')


def test_dump_data_with_valid_params():
    token_id = 1
    data_id = 2
    golden_data = torch.tensor([1])
    my_path = "test_my_path"
    output_path = "./output"
    # Mock external functions and dependencies
    with mock.patch('components.llm.msit_llm.dump.manual_dump.load_file_to_read_common_check', 
        return_value=my_path) as mocked_load, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.check_output_path_legality') as mocked_check, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.ms_makedirs'), \
        mock.patch('components.llm.msit_llm.dump.manual_dump.get_ait_dump_path', return_value='ait_dump'), \
        mock.patch('os.path.exists', return_value=False), \
        mock.patch('torch.save'), \
        mock.patch('components.llm.msit_llm.dump.manual_dump.write_json_file') as mocked_write:
        dump_data(token_id, data_id, golden_data, my_path, output_path)
        # Ensure that write_json_file was called with the correct arguments
        cur_pid = os.getpid()
        device_id = golden_data.get_device() if golden_data.is_cuda else -1
        output_path_prefix = os.path.join(output_path, 'ait_dump', f"{cur_pid}_{device_id}")
        golden_data_dir = os.path.join(output_path_prefix, "golden_tensor", str(token_id))
        golden_data_path = os.path.join(golden_data_dir, f'{data_id}_tensor.pth')
        json_path = os.path.join(output_path_prefix, "golden_tensor", "metadata.json")
        mocked_write.assert_called_once_with(data_id, golden_data_path, json_path, token_id, my_path)


def test_dump_data_valid_params_cpu_tensor():
    token_id = 1
    data_id = 2
    golden_data = torch.tensor([1])
    my_path = "test_my_path"
    output_path = "./output"
    # Mock external functions and dependencies
    with mock.patch('components.llm.msit_llm.dump.manual_dump.load_file_to_read_common_check', 
        return_value=my_path) as mocked_load, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.check_output_path_legality') as mocked_check, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.ms_makedirs') as mocked_makedirs, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.get_ait_dump_path', return_value='ait_dump'), \
        mock.patch('os.path.exists', side_effect=lambda x: False if 'golden_tensor' in x else True), \
        mock.patch('torch.save') as mocked_torch_save, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.write_json_file') as mocked_write:
        dump_data(token_id, data_id, golden_data, my_path, output_path)
        # Ensure that ms_makedirs was called to create directories
        cur_pid = os.getpid()
        device_id = -1  # CPU tensor
        output_path_prefix = os.path.join(output_path, 'ait_dump', f"{cur_pid}_{device_id}")
        golden_data_dir = os.path.join(output_path_prefix, "golden_tensor", str(token_id))
        mocked_makedirs.assert_called_once_with(golden_data_dir)
        # Ensure that torch.save was called with the correct arguments
        golden_data_path = os.path.join(golden_data_dir, f'{data_id}_tensor.pth')
        mocked_torch_save.assert_called_once_with(golden_data, golden_data_path)
        # Ensure that write_json_file was called with the correct arguments
        json_path = os.path.join(output_path_prefix, "golden_tensor", "metadata.json")
        mocked_write.assert_called_once_with(data_id, golden_data_path, json_path, token_id, my_path)


def test_dump_data_existing_directories():
    token_id = 1
    data_id = 2
    golden_data = torch.tensor([1])
    my_path = "test_my_path"
    output_path = "./output"
    # Mock external functions and dependencies
    with mock.patch('components.llm.msit_llm.dump.manual_dump.load_file_to_read_common_check', 
        return_value=my_path) as mocked_load, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.check_output_path_legality') as mocked_check, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.ms_makedirs') as mocked_makedirs, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.get_ait_dump_path', return_value='ait_dump'), \
        mock.patch('os.path.exists', return_value=True), \
        mock.patch('torch.save') as mocked_torch_save, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.write_json_file') as mocked_write:
        dump_data(token_id, data_id, golden_data, my_path, output_path)
        # Ensure that ms_makedirs was not called because directories already exist
        mocked_makedirs.assert_not_called()
        # Ensure that torch.save and write_json_file were still called correctly
        cur_pid = os.getpid()
        device_id = -1  # CPU tensor
        output_path_prefix = os.path.join(output_path, 'ait_dump', f"{cur_pid}_{device_id}")
        golden_data_dir = os.path.join(output_path_prefix, "golden_tensor", str(token_id))
        golden_data_path = os.path.join(golden_data_dir, f'{data_id}_tensor.pth')
        json_path = os.path.join(output_path_prefix, "golden_tensor", "metadata.json")
        mocked_torch_save.assert_called_once_with(golden_data, golden_data_path)
        mocked_write.assert_called_once_with(data_id, golden_data_path, json_path, token_id, my_path)


def test_dump_data_check_output_path_legality_fails():
    token_id = 1
    data_id = 2
    golden_data = torch.tensor([1])
    my_path = "test_my_path"
    output_path = "./invalid_output_path"
    # Mock external functions and dependencies
    with mock.patch('components.llm.msit_llm.dump.manual_dump.load_file_to_read_common_check', 
    return_value=my_path) as mocked_load, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.check_output_path_legality', 
        side_effect=ValueError("Invalid output path")) as mocked_check, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.ms_makedirs') as mocked_makedirs, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.get_ait_dump_path', return_value='ait_dump'), \
        mock.patch('os.path.exists', return_value=True), \
        mock.patch('torch.save') as mocked_torch_save, \
        mock.patch('components.llm.msit_llm.dump.manual_dump.write_json_file') as mocked_write:
        # Call the function under test
        with pytest.raises(ValueError, match="Invalid output path"):
            dump_data(token_id, data_id, golden_data, my_path, output_path)
        # Ensure no further calls were made due to the failed legality check
        mocked_makedirs.assert_not_called()

@patch("json.load")
@patch("json.dump")
@patch('os.path.exists', return_value=True)
@patch('components.llm.msit_llm.dump.manual_dump.load_file_to_read_common_check')
@patch('components.llm.msit_llm.dump.manual_dump.ms_open')
def test_write_json_file_given_valid_parameters_then_pass(
        mock_ms_open, mock_load_file_to_read_common_check, mock_os, mock_json_dump, mock_json_load
    ):
        data_id = "test_data_id"
        data_path = "test_data_path"
        json_path = "test_json_path"
        token_id = "test_token_id"
        my_path = "test_my_path"
        write_json_file(data_id, data_path, json_path, token_id, my_path)
        mock_os.assert_called_with(json_path)