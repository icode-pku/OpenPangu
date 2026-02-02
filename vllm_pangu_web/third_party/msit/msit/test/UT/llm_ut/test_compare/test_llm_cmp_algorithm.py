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
import os.path

import pytest
import torch

from components.utils.cmp_algorithm import cosine_similarity, max_relative_error, mean_relative_error, \
    relative_euclidean_distance, l1_norm


@pytest.fixture(scope='module', autouse=True)
def golden_data():
    golden_data = torch.ones((2, 3))
    yield golden_data


@pytest.fixture(scope='module', autouse=True)
def test_data():
    test_data = torch.ones((2, 3))
    yield test_data


def test_cosine_similarity(golden_data, test_data):
    res, message = cosine_similarity(golden_data.reshape(-1), test_data.reshape(-1))
    assert res == 1.0
    assert message == ''


def test_max_relative_error(golden_data, test_data):
    res, message = max_relative_error(golden_data, test_data)
    assert res == 0.0
    assert message == ''


def test_mean_relative_error(golden_data, test_data):
    res, message = mean_relative_error(golden_data, test_data)
    assert res == 0.0
    assert message == ''


def test_relative_euclidean_distance(golden_data, test_data):
    res, message = relative_euclidean_distance(golden_data, test_data)
    assert res == 0.0
    assert message == ''


def test_relative_euclidean_distance_when_low_acc(golden_data, test_data):
    test_data = 10 * test_data
    res, message = relative_euclidean_distance(golden_data, test_data)
    assert res == 9.0
    assert message == ''

def test_l1_norm(golden_data, test_data):
    res, message = l1_norm(golden_data, test_data)
    assert res == 0.0
    assert message == ''