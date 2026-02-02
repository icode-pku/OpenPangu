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

import os, sys
import subprocess
import shutil

from unittest import mock
import pytest
import numpy as np
import pandas as pd

from auto_optimizer import OnnxGraph
from auto_optimizer.graph_refactor.onnx import OnnxNode, OnnxPlaceHolder, OnnxInitializer
from msquickcmp.common import utils
from msquickcmp.common.utils import AccuracyCompareException


@pytest.fixture(scope="function", autouse=True)
def import_cmp_process(monkeypatch):
    mock_acl = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "acl", mock_acl)
    from msquickcmp.single_op import single_op as sp
    return sp


@pytest.fixture(scope="function")
def create_broken_graph(name: str = 'test_broken'):
    input_ = OnnxPlaceHolder('input', np.dtype('float32'), [1, 3, 224, 224])
    output = OnnxPlaceHolder('output', np.dtype('float32'), [1, 3, 224, 224])
    node_0 = OnnxNode('sqrt0', 'Sqrt', inputs=['input'], outputs=['sqrt0_output'], attrs={})
    node_1 = OnnxNode('relu1', 'Relu', inputs=['sqrt0_output'], outputs=['relu1_output'], attrs={})
    node_2 = OnnxNode('sqrt2', 'Sqrt', inputs=['relu1_output'], outputs=['sqrt2_output'], attrs={})
    node_3 = OnnxNode('relu3', 'Relu', inputs=['sqrt2_output'], outputs=['relu3_output'], attrs={})
    node_4 = OnnxNode('flatten4', 'Flatten', inputs=['relu3_output'], outputs=['flatten4_output'], attrs={})
    return OnnxGraph(
        name=name,
        nodes=[node_0, node_1, node_2, node_3, node_4],
        inputs=[input_],
        outputs=[output],
    )


@pytest.fixture(scope="function")
def create_dynamic_divide_onnx_graph(name: str = 'test_dynamic_divide_onnx'):
    input_1 = OnnxPlaceHolder('input', np.dtype('float32'), [8, 3, 768, 768])
    input_2 = OnnxPlaceHolder('sqrt0_output', np.dtype('float32'), [8, 3, 768, 768])
    output_1 = OnnxPlaceHolder('out_0_sqrt0_output', np.dtype('float32'), [8, 3, 768, 768])
    node_0 = OnnxNode('sqrt0', 'Sqrt', inputs=['input'], outputs=['out_0_sqrt0_output'], attrs={})
    node_1 = OnnxNode('relu1', 'Relu', inputs=['sqrt0_output'], outputs=['out_1_relu1_output'], attrs={})
    return OnnxGraph(
        name=name,
        nodes=[node_0, node_1],
        inputs=[input_1, input_2],
        outputs=[output_1],
    )


@pytest.fixture(scope="function")
def create_accumulate_shape_size_graph(name: str = 'test_accumulate_shape_size'):
    input_ = OnnxPlaceHolder('input', np.dtype('float32'), [1, 3, 224, 224])
    output = OnnxPlaceHolder('output', np.dtype('float32'), [1, 3, 224, 224])
    input_1 = OnnxPlaceHolder('input_1', np.dtype('float32'), [8, 3, 768, 768])
    output_1 = OnnxPlaceHolder('output_1', np.dtype('float32'), [8, 3, 768, 768])
    node_0 = OnnxNode('sqrt0', 'Sqrt', inputs=['input'], outputs=['output'], attrs={})
    node_1 = OnnxNode('sqrt1', 'Sqrt', inputs=['input_1'], outputs=['output_1'], attrs={})
    return OnnxGraph(
        name=name,
        nodes=[node_0, node_1],
        inputs=[input_, input_1],
        outputs=[output, output_1],
    )


def test_broken_when_valid_then_passs(create_broken_graph, import_cmp_process):
    sp = import_cmp_process
    create_broken_graph.infer_shape()
    subgraph_onnx_file = './broken.onnx'
    sp.broken(create_broken_graph, subgraph_onnx_file)
    os.remove(subgraph_onnx_file)
    assert len(create_broken_graph.inputs) == 5
    assert len(create_broken_graph.outputs) == 6


def test_dynamic_divide_onnx_when_valid_then_pass(create_dynamic_divide_onnx_graph, import_cmp_process):
    sp = import_cmp_process
    create_dynamic_divide_onnx_graph.infer_shape()
    out_path = './test_dynamic_divide_onnx/'
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path, 0o750)
    memory_size = 2 * 8 * 3 * 768 * 768
    subonnx_list = sp.dynamic_divide_onnx(out_path, create_dynamic_divide_onnx_graph, memory_size)
    shutil.rmtree(out_path)
    assert subonnx_list == ['./test_dynamic_divide_onnx/0_broken.onnx', './test_dynamic_divide_onnx/1_broken.onnx']


def test_accumulate_shape_size_when_valid_then_pass(create_accumulate_shape_size_graph, import_cmp_process):
    sp = import_cmp_process
    create_accumulate_shape_size_graph.infer_shape()
    node_0 = OnnxNode('sqrt0', 'Sqrt', inputs=['input'], outputs=['output'], attrs={})
    node_1 = OnnxNode('sqrt1', 'Sqrt', inputs=['input_1'], outputs=['output_1'], attrs={})
    ans_1 = sp.accumulate_shape_size(node_0, create_accumulate_shape_size_graph)
    ans_2 = sp.accumulate_shape_size(node_1, create_accumulate_shape_size_graph)
    assert ans_1 == np.dtype('float32').itemsize * 2 * 1 * 3 * 224 * 224
    assert ans_2 == np.dtype('float32').itemsize * 2 * 8 * 3 * 768 * 768


def test_generate_single_op_dir_when_valid_then_pass(import_cmp_process):
    sp = import_cmp_process
    out_path = 'fake_test_path'
    single_op_dir = sp.generate_single_op_dir(out_path)
    shutil.rmtree(out_path)
    assert single_op_dir == 'fake_test_path/single_op'


def test_get_memory_size_by_soc_type_when_invalid_npu_id_then_failed(import_cmp_process):
    sp = import_cmp_process
    with pytest.raises(AccuracyCompareException):
        with mock.patch("subprocess.run", return_value=subprocess.CompletedProcess(
                        args=[''],
                        returncode=0,
                        stdout=b''
                        )):
            ret = sp.get_memory_size_by_soc_type(0)
            assert ret == utils.ACCURACY_COMPARISON_INVALID_DEVICE_ERROR


def test_get_memory_size_by_soc_type_when_invalid_memory_size_then_failed(import_cmp_process):
    sp = import_cmp_process
    with pytest.raises(AccuracyCompareException):
        with mock.patch("subprocess.run", return_value=subprocess.CompletedProcess(
                        args=[''],
                        returncode=0,
                        stdout=b'NPU ID 1\nDDR Capacity(MB) -1\n'
                        )):
            ret = sp.get_memory_size_by_soc_type(0)
            assert ret == utils.ACCURACY_COMPARISON_INVALID_DEVICE_ERROR


def test_find_all_csv_when_valid_then_pass(import_cmp_process):
    sp = import_cmp_process
    out_path = 'find_all_csv_test_path'
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path, 0o750)
    df1 = pd.DataFrame({'sum' : [0, 0, 0], 'name' : ['0', '0', '0']})
    df1_path = os.path.join(out_path, 'file1.csv')
    df1.to_csv(df1_path, index=False)
    df2 = pd.DataFrame({'value' : [1, 2, 3], 'name' : ['a', 'b', 'c']})
    df2_path = os.path.join(out_path, 'file2.csv')
    df2.to_csv(df2_path, index=False)
    all_csv_list = sp.find_all_csv(out_path)
    shutil.rmtree(out_path)
    all_csv_list.sort()
    assert all_csv_list == ['find_all_csv_test_path/file1.csv', 'find_all_csv_test_path/file2.csv']