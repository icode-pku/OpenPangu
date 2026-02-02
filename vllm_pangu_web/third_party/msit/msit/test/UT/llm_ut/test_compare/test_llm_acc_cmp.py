import os
import shutil
import json
import stat
from unittest.mock import patch

import pytest
import torch
import numpy as np

from components.llm.msit_llm.common.constant import GLOBAL_AIT_DUMP_PATH
from components.utils.file_utils import FileChecker
import msit_llm.compare.cmp_utils

FILE_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
FAKE_GOLDEN_DATA_PATH = "test_acc_cmp_fake_golden_data.npy"
FAKE_MY_DATA_PATH = "test_acc_cmp_fake_test_data.npy"

ori_file_common_check = FileChecker.common_check


def mock_file_common_check():
    def common_check(self):
        pass
    setattr(FileChecker, 'common_check', common_check)


def recover_file_common_check():
    setattr(FileChecker, 'common_check', ori_file_common_check)


@pytest.fixture(scope="function")
def golden_data_file():
    golden_data = np.ones((2, 3)).astype(np.float32)
    golden_data[0][0] = 10
    np.save(FAKE_GOLDEN_DATA_PATH, golden_data)

    yield FAKE_GOLDEN_DATA_PATH

    if os.path.exists(FAKE_GOLDEN_DATA_PATH):
        os.remove(FAKE_GOLDEN_DATA_PATH)


@pytest.fixture(scope="function")
def test_data_file():
    test_data = np.ones((2, 3)).astype(np.float32)
    test_data[0][0] = 10
    np.save(FAKE_MY_DATA_PATH, test_data)

    yield FAKE_MY_DATA_PATH

    if os.path.exists(FAKE_MY_DATA_PATH):
        os.remove(FAKE_MY_DATA_PATH)


@pytest.fixture(scope="function")
def test_dat_path():
    test_data_path = "test_acc_cmp_fake_test_data.dat"  # No need to create actual file

    yield test_data_path

    if os.path.exists(test_data_path):
        os.remove(test_data_path)


@pytest.fixture(scope="function")
def test_metadata_path():
    test_metadata_path = "test_acc_cmp_fake_metadata"
    if not os.path.exists(test_metadata_path):
        os.makedirs(test_metadata_path, mode=0o750)
    metadata = {0: {0: [FAKE_GOLDEN_DATA_PATH, FAKE_MY_DATA_PATH]}}

    metadata_json_path = os.path.join(test_metadata_path, "metadata.json")
    with os.fdopen(os.open(metadata_json_path, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), "w") as ff:
        json.dump(metadata, ff)

    yield test_metadata_path

    if os.path.exists(test_metadata_path):
        shutil.rmtree(test_metadata_path)


@pytest.fixture(scope="function")
def test_torch_path():
    test_torch_path = "test_acc_cmp_fake_torch"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    resource_dir = os.path.join(cur_dir, '..', '..', 'resource')
    with open(os.path.join(resource_dir, "llm", "compare", "model_tree.json")) as f:
        torch_topo = json.load(f)

    if not os.path.exists(test_torch_path):
        os.makedirs(os.path.join(test_torch_path, "1111_npu0/0/"), mode=0o750)
        _json_path = os.path.join(test_torch_path, "1111_npu0/model_tree.json")
        with os.fdopen(os.open(_json_path, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), "w") as ff:
            json.dump(torch_topo, ff)

    yield test_torch_path

    if os.path.exists(test_torch_path):
        shutil.rmtree(test_torch_path)


@pytest.fixture(scope="function")
def test_atb_path():
    test_atb_path = "test_acc_cmp_fake_atb"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    resource_dir = os.path.join(cur_dir, '..', '..', 'resource')
    with open(os.path.join(resource_dir, "llm", "compare", "Bloom7BFlashAttentionModel.json")) as f:
        atb_topo = json.load(f)

    if not os.path.exists(test_atb_path):
        os.makedirs(os.path.join(test_atb_path, f"{GLOBAL_AIT_DUMP_PATH}/tensors/1_2222/0/"), mode=0o750)
        os.makedirs(os.path.join(test_atb_path, f"{GLOBAL_AIT_DUMP_PATH}/model/2222/"), mode=0o750)
        _json_path = os.path.join(test_atb_path, f"{GLOBAL_AIT_DUMP_PATH}/model/2222/BloomModel.json")
        with os.fdopen(os.open(_json_path, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), "w") as ff:
            json.dump(atb_topo, ff)

    yield test_atb_path

    if os.path.exists(test_atb_path):
        shutil.rmtree(test_atb_path)


def test_check_tensor_given_golden_data_when_nan_then_false():
    result, message = msit_llm.compare.cmp_utils.check_tensor(
        torch.zeros([2]).float() + torch.nan, torch.zeros([2]).float()
    )
    assert result is False and len(message) > 0 and "golden" in message.lower()


def test_fill_row_data_given_my_path_when_valid_then_pass(golden_data_file, test_data_file):
    mock_file_common_check()
    data_info = msit_llm.compare.cmp_utils.BasicDataInfo(golden_data_file, test_data_file, 0, 0)
    recover_file_common_check()
    row_data = msit_llm.compare.cmp_utils.fill_row_data(data_info)
    assert isinstance(row_data, dict) and len(row_data) == 24
    assert row_data["cosine_similarity"] == 1.0
    assert len(row_data["cmp_fail_reason"]) == 0


def test_fill_row_data_given_loaded_my_data_when_valid_then_pass(golden_data_file):
    golden_data = np.load(golden_data_file)
    loaded_my_data = np.zeros_like(golden_data)
    mock_file_common_check()
    data_info = msit_llm.compare.cmp_utils.BasicDataInfo(golden_data_file, "test")
    recover_file_common_check()
    row_data = msit_llm.compare.cmp_utils.fill_row_data(data_info, loaded_my_data=loaded_my_data)
    assert isinstance(row_data, dict) and len(row_data) == 24
    assert row_data["cosine_similarity"] == 0.0


def test_fill_row_data_given_my_path_when_dir_then_error(golden_data_file):
    mock_file_common_check()
    data_info = msit_llm.compare.cmp_utils.BasicDataInfo(golden_data_file, "/")
    recover_file_common_check()
    row_data = msit_llm.compare.cmp_utils.fill_row_data(data_info)
    assert isinstance(row_data, dict) and len(row_data) == 7
    assert len(row_data["cmp_fail_reason"]) > 0


def test_fill_row_data_given_golden_data_path_when_empty_then_error(test_data_file):
    mock_file_common_check()
    data_info = msit_llm.compare.cmp_utils.BasicDataInfo("", test_data_file)
    recover_file_common_check()
    row_data = msit_llm.compare.cmp_utils.fill_row_data(data_info)
    assert isinstance(row_data, dict) and len(row_data) == 7
    assert len(row_data["cmp_fail_reason"]) > 0


def test_fill_row_data_given_my_path_when_nan_then_error(golden_data_file):
    golden_data = np.load(golden_data_file)
    loaded_my_data = np.zeros_like(golden_data) + np.nan
    mock_file_common_check()
    data_info = msit_llm.compare.cmp_utils.BasicDataInfo(golden_data_file, "test")
    recover_file_common_check()
    row_data = msit_llm.compare.cmp_utils.fill_row_data(data_info, loaded_my_data=loaded_my_data)
    assert isinstance(row_data, dict) and len(row_data) == 17
    assert len(row_data["cmp_fail_reason"]) > 0


def test_fill_row_data_given_my_path_when_shape_not_match_then_error(golden_data_file):
    golden_data = np.load(golden_data_file)
    loaded_my_data = np.zeros([])
    mock_file_common_check()
    data_info = msit_llm.compare.cmp_utils.BasicDataInfo(golden_data_file, "test")
    recover_file_common_check()
    row_data = msit_llm.compare.cmp_utils.fill_row_data(data_info, loaded_my_data=loaded_my_data)
    assert isinstance(row_data, dict) and len(row_data) == 17
    assert len(row_data["cmp_fail_reason"]) > 0


def test_read_data_given_data_file_when_valid_npy_then_pass(golden_data_file, test_data_file):
    data = msit_llm.compare.cmp_utils.read_data(test_data_file)
    golden = msit_llm.compare.cmp_utils.read_data(golden_data_file)
    assert (data == golden).all()


def test_read_data_when_npy(golden_data_file, test_data_file):
    data = msit_llm.compare.cmp_utils.read_data(test_data_file)
    golden = torch.tensor(np.load(golden_data_file))
    assert torch.all(data == golden).item()


def test_read_data_given_data_file_when_invalid_type_then_error(test_dat_path):
    with pytest.raises(FileNotFoundError):
        msit_llm.compare.cmp_utils.read_data(test_dat_path)


def test_compare_data_given_data_file_when_valid_then_pass(golden_data_file, test_data_file):
    test_data = msit_llm.compare.cmp_utils.read_data(test_data_file)
    golden_data = msit_llm.compare.cmp_utils.read_data(golden_data_file)
    res = msit_llm.compare.cmp_utils.compare_data(test_data, golden_data)
    assert res == {
        "cosine_similarity": 1.0,
        "max_relative_error": 0.0,
        "mean_relative_error": 0.0,
        "kl_divergence": 0.0,
        "max_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "relative_euclidean_distance": 0.0,
        "cmp_fail_reason": "",
    }