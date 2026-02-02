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
import pytest

from msquickcmp.net_compare import analyser


OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
OPEN_MODES = stat.S_IWUSR | stat.S_IRUSR


@pytest.fixture(scope="module", autouse=True)
def fake_csv_file():
    test_csv_file_name = "./fake_test_csv.csv"

    columns = ["Index", "OpType", "NPUDump", "DataType", "GroundTruth"]
    columns += ["CosineSimilarity", "RelativeEuclideanDistance", "KullbackLeiblerDivergence"]
    columns += ["RootMeanSquareError", "MaxRelativeError", "MeanRelativeError"]

    data = [
        "213,Const,dynamic_const_7221_361,NaN,*,NaN,NaN,NaN,NaN,NaN,NaN",
        "214,Mul,Mul_6,float16,Mul_6,1,0,0,0,NaN,NaN",
        "214,Mul,Mul_6,NaN,Mul_6,NaN,NaN,NaN,NaN,NaN",
        "214,Mul,Mul_6,float16,Mul_6,0.672178,1,0,inf,inf,inf",
        "241,ArgMaxV2,ArgMax_1180,float32,ArgMax_1180,0.905575,0.429081,inf,4.061347,5254036,2.989594",
        "241,ArgMaxV2,ArgMax_1180,NaN,ArgMax_1180,NaN,NaN,NaN,NaN,NaN,NaN",
        "241,ArgMaxV2,ArgMax_1180,NaN,ArgMax_1180,NaN,NaN,NaN,NaN,NaN,NaN",
        "316,BatchMatMulV2,PartitionedCall_ArgMax_1180_MatMul_179,NaN,MatMul_179,NaN,NaN,NaN,NaN,NaN,NaN",
        "316,BatchMatMulV2,PartitionedCall_ArgMax_1180_MatMul_179,NaN,MatMul_179,NaN,NaN,NaN,NaN,NaN,NaN",
        "316,BatchMatMulV2,MatMul_179,float16,MatMul_179,0.434028,0.950848,0.01125,0.276453,268562.06,5.7565",
        "953,Add,Add_1179,float16,Add_1179,0.905575,0.429081,inf,4.061347,5254036,2.989594",
        "955,NaN,Node_Output,NaN,output_0.npy,0.704,0.761698,inf,3419.279248,63.709677,1.655235",
    ]

    with os.fdopen(os.open(test_csv_file_name, OPEN_FLAGS, OPEN_MODES), 'w') as fout:
        fout.write(",".join(columns) + "\n")
        fout.write("\n".join(data))

    yield test_csv_file_name

    if os.path.exists(test_csv_file_name):
        os.remove(test_csv_file_name)


def test_analyser_init_given_valid_file_when_any_then_pass(fake_csv_file):
    aa = analyser.Analyser(fake_csv_file)


def test_analyser_init_given_valid_folder_when_any_then_pass(fake_csv_file):
    aa = analyser.Analyser(os.path.dirname(fake_csv_file))


def test_analyser_init_given_invalid_when_any_then_fail():
    with pytest.raises(TypeError):
        aa = analyser.Analyser(42)

    with pytest.raises(ValueError):
        aa = analyser.Analyser("foo.foo")

    with pytest.raises(OSError):
        aa = analyser.Analyser("not_exists_test_csv.csv")


def test_analyser_call_given_valid_when_overall_then_pass(fake_csv_file):
    aa = analyser.Analyser(fake_csv_file)
    results, monitors = aa()

    assert len(results) == 1
    assert results[0]["Index"] == "214"
    assert monitors == [["CosineSimilarity", "RelativeEuclideanDistance"]]


def test_analyser_call_given_valid_when_eash_then_pass(fake_csv_file):
    aa = analyser.Analyser(fake_csv_file)
    results, monitors = aa(strategy=analyser.STRATEGIES.FIRST_INVALID_EACH)

    assert len(results) == 3
    assert len(monitors) == 3
    assert [ii["Index"] for ii in results] == ["214", "241", "316"]
    assert monitors[0] == ["CosineSimilarity", "RelativeEuclideanDistance"]
    assert monitors[1] == ["RootMeanSquareError", "MeanRelativeError"]
    assert monitors[2] == ["KullbackLeiblerDivergence"]


def test_analyser_call_strategy_given_invalid_when_any_then_fail(fake_csv_file):
    aa = analyser.Analyser(fake_csv_file)
    with pytest.raises(ValueError):
        aa(strategy=42)


def test_analyser_call_max_column_len_given_invalid_when_any_then_fail(fake_csv_file):
    aa = analyser.Analyser(fake_csv_file)
    with pytest.raises(TypeError):
        aa(max_column_len="foo")

    with pytest.raises(ValueError):
        aa(max_column_len=0)
