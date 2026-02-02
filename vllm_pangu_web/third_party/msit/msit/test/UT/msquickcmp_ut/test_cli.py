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

import pytest
from unittest.mock import patch, MagicMock
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.common import utils


@pytest.fixture(scope="module", autouse=True)
def compare_cli() -> None:
    cmp_args = CmpArgsAdapter(
        gold_model="./fake.onnx",
        om_model="./fake.om",
        input_data_path="",
        out_path="",
        input_shape="",
        device='0',
        output_size="",
        output_nodes="",
        advisor=False,
        dym_shape_range="",
        dump=True,
        bin2npy=False,
    )
    yield cmp_args


def test_args_invalid_path_err(compare_cli):
    mock_acl = MagicMock()
    with pytest.raises(utils.AccuracyCompareException) as error, \
         patch.dict('sys.modules', {
             'acl': mock_acl,
         }):
        from msquickcmp.cmp_process import cmp_process
        cmp_process(compare_cli, True)

    assert error.value.error_info == utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR
