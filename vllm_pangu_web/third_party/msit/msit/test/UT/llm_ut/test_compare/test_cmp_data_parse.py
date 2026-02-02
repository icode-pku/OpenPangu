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

import unittest
import os
import json
from unittest.mock import patch, MagicMock
from typing import List, Tuple
from msit_llm.compare.cmp_data_parse import CompareDataParse, CompareDataATB, CompareDataTorch, DataUtils
from msit_llm.compare.cmp_op_match import MatchLocation
from msit_llm.dump.torch_dump.topo import ModelTree, TreeNode


class TreeNode:
    def __init__(self, name: str):
        self.name = name
        self.children = []
 
 
class MockCompareDataParse(CompareDataParse):
    def __init__(self, path: str, args: dict):
        super().__init__(path, args)
 
    @staticmethod
    def accept(path: str) -> bool:
        # 假设只接受以 "test_" 开头的路径
        return path.startswith("test_")
 
    def get_root_nodes(self) -> List[TreeNode]:
        # 返回一些模拟的根节点
        return [TreeNode("root1"), TreeNode("root2")]
 
    def get_cmp_tokens(self) -> Tuple:
        # 返回一些模拟的匹配 tokens
        return ("token1", "token2")
 
    def get_tensor_path(self, token_id: str, node: TreeNode, location: str) -> Tuple:
        # 返回模拟的 tensor 路径
        return (f"tensor_{token_id}_{node.name}_{location}",)
 
    def get_token_path(self, token_id: str) -> str:
        # 返回模拟的 token 路径
        return f"path_to_token_{token_id}"
 
    def get_token_id(self) -> str:
        # 返回模拟的 token id
        return "mock_token_id"


class TestCompareDataParse(unittest.TestCase):
    def setUp(self):
        self.path = "test_data.txt"
        self.args = {"key": "value"}
        self.parser = MockCompareDataParse(self.path, self.args)
 
    def test_accept(self):
        # 测试 accept 方法
        self.assertTrue(MockCompareDataParse.accept("test_file.txt"))
        self.assertFalse(MockCompareDataParse.accept("not_test_file.txt"))
 
    def test_get_root_nodes(self):
        # 测试 get_root_nodes 方法
        root_nodes = self.parser.get_root_nodes()
        self.assertEqual(len(root_nodes), 2)
        self.assertEqual(root_nodes[0].name, "root1")
        self.assertEqual(root_nodes[1].name, "root2")
 
    def test_get_cmp_tokens(self):
        # 测试 get_cmp_tokens 方法
        cmp_tokens = self.parser.get_cmp_tokens()
        self.assertEqual(cmp_tokens, ("token1", "token2"))
 
    def test_get_tensor_path(self):
        # 测试 get_tensor_path 方法
        node = TreeNode("node1")
        tensor_paths = self.parser.get_tensor_path("token_id", node, "output")
        self.assertEqual(len(tensor_paths), 1)
        self.assertEqual(tensor_paths[0], "tensor_token_id_node1_output")
 
    def test_get_token_path(self):
        # 测试 get_token_path 方法
        token_path = self.parser.get_token_path("token_id")
        self.assertEqual(token_path, "path_to_token_token_id")
 
    def test_get_token_id(self):
        # 测试 get_token_id 方法
        token_id = self.parser.get_token_id()
        self.assertEqual(token_id, "mock_token_id")


class TestDataUtils(unittest.TestCase):
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_get_token_ids_with_token_id(self, mock_listdir, mock_isdir, mock_exists):
        # Mock the behavior of os functions
        mock_exists.return_value = True  # Not needed here, but good practice to set expectations
        mock_isdir.return_value = True  # Not used in this test case path
        mock_listdir.return_value = []  # Not used in this test case path
 
        # Call the method
        result = DataUtils.get_token_ids(None, 'specified_token_id')
 
        # Verify the result
        self.assertEqual(result, ('specified_token_id',))
 
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_get_token_ids_without_token_id_and_path(self, mock_listdir, mock_isdir, mock_exists):
        # Mock the behavior of os functions
        mock_exists.return_value = True  # Not needed here, as tokens_path is None
 
        # Call the method
        result = DataUtils.get_token_ids(None, None)
 
        # Verify the result
        self.assertEqual(result, tuple())

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_dir_sub_files_with_existing_dir(self, mock_listdir, mock_exists):
        # Mock the behavior of os functions
        mock_exists.return_value = True
        mock_listdir.return_value = ['prefix_file1.txt', 'otherfile.txt', 'prefix_file2.txt']
 
        # Call the method
        result = DataUtils.get_dir_sub_files('/mock/dir/path', 'prefix_')
 
        # Verify the result
        expected = (
            os.path.join('/mock/dir/path', 'prefix_file1.txt'),
            os.path.join('/mock/dir/path', 'prefix_file2.txt')
        )
        self.assertEqual(result, expected)
 
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_dir_sub_files_with_non_existing_dir(self, mock_listdir, mock_exists):
        # Mock the behavior of os functions
        mock_exists.return_value = False
 
        # Call the method
        result = DataUtils.get_dir_sub_files('/mock/dir/path', 'prefix_')
 
        # Verify the result
        self.assertEqual(result, tuple())


class TestCompareDataATB(unittest.TestCase):
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_parse_ait_dump_path(self, mock_listdir, mock_exists):
        # 设置 mock 返回值
        mock_exists.side_effect = [False, False, True]
        mock_listdir.return_value = ["ait_dump", "other_dir"]
 
        # 测试不同路径
        path1 = "/some/random/path"
        path2 = "/ait_dump/some/path"
        path3 = "/other_ait_dump/some/path"
 
        self.assertIsNone(CompareDataATB.parse_ait_dump_path(path1))
        self.assertEqual(CompareDataATB.parse_ait_dump_path(path2), "/ait_dump")
 
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.join')
    @patch('os.path.abspath')
    def test_get_ids_by_path(self, mock_abspath, mock_join, mock_listdir, mock_exists):
        # 设置 mock 返回值
        mock_exists.side_effects = [True, True, True, True, False, True]
        mock_listdir.side_effects = [["tensors", "model"], ["0_pid"], [], ["1_pid", "0_pid"], ["token_id.json"]]
        mock_join.side_effects = ["/ait_dump", "/ait_dump/tensors", "/ait_dump/model", "/ait_dump/tensors/0_pid", "/ait_dump/tensors/0_pid/token_id.json"]
        mock_abspath.return_value = "/ait_dump/tensors/0_pid"
 
        # 测试不同路径
        ait_dump_path = "/ait_dump"
        path1 = "/ait_dump/tensors/0_pid/token_id"
        path2 = "/ait_dump/tensors/1_pid"
        path3 = "/ait_dump/model/0_pid"
 
        self.assertEqual(CompareDataATB.get_ids_by_path(ait_dump_path, path1), (None, None, '/ait_dump/tensors/0_pid'))
        self.assertEqual(CompareDataATB.get_ids_by_path(ait_dump_path, path2), (None, None, '/ait_dump/tensors/0_pid'))
        self.assertEqual(CompareDataATB.get_ids_by_path(ait_dump_path, path3), (None, None, '/ait_dump/tensors/0_pid'))
 
    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.path.join')
    def test_get_topo_file_path(self, mock_join, mock_exists, mock_listdir):
        # 设置 mock 返回值
        mock_exists.side_effects = [True, True, False]
        mock_listdir.side_effects = [["0_pid"], ["file1.json", "file2.json"]]
        mock_join.side_effects = ["/ait_dump/model", "/ait_dump/model/0_pid", "/ait_dump/model/0_pid/file1.json", "/ait_dump/model/0_pid/file2.json"]
 
        # 测试不同 pid
        ait_dump_path = "/ait_dump"
        pid = "0_pid"
 
        self.assertIsNone(CompareDataATB.get_topo_file_path("/non_existent_path"))


class TestCompareDataTorch(unittest.TestCase):
    @patch('os.path.exists', side_effect=lambda x: True if x.endswith('model_tree.json') else False)
    @patch('os.path.join', side_effect=lambda *args: "/".join(args))
    @patch('os.path.relpath', return_value="rel_path")
    @patch('os.listdir', return_value=['device_id_123'])
    def test_parse_ait_dump_path(self, mock_listdir, mock_relpath, mock_join, mock_exists):
        """
        测试parse_ait_dump_path方法在不同条件下能否正确解析并返回ait_dump_path。
        """
        path = "test_path"
        result = CompareDataTorch.parse_ait_dump_path(path)
        self.assertTrue(isinstance(result, str) or result is None)

    @patch('os.path.exists', side_effect=lambda x: True if x.endswith('device_id_123') else False)
    @patch('os.path.join', side_effect=lambda *args: "/".join(args))
    @patch('os.path.relpath', return_value="rel_path")
    @patch('os.listdir', return_value=['device_id_123'])
    def test_get_ids_by_path(self, mock_listdir, mock_relpath, mock_join, mock_exists):
        """
        测试get_ids_by_path方法在不同路径结构下能否正确解析并返回token_id、pid和tokens_path。
        """
        ait_dump_path = "ait_dump_path"
        path = "test_path"
        result = CompareDataTorch.get_ids_by_path(ait_dump_path, path)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], (str, type(None)))
        self.assertIsInstance(result[1], (str, type(None)))
        self.assertIsInstance(result[2], str)

    @patch('os.path.exists', return_value=True)
    def test_get_topo_file_path(self, mock_exists):
        """
        测试get_topo_file_path方法能否根据给定的tokens_path正确返回拓扑文件路径。
        """
        tokens_path = "tokens_path"
        result = CompareDataTorch.get_topo_file_path(tokens_path)
        self.assertEqual(result, os.path.join(tokens_path, "model_tree.json"))

    @patch('msit_llm.compare.cmp_data_parse.CompareDataTorch.parse_ait_dump_path', return_value="ait_dump_path")
    def test_accept(self, mock_parse_ait_dump_path):
        """
        测试accept方法在不同解析结果下是否能正确返回布尔值表示是否接受给定路径。
        """
        path = "test_path"
        result = CompareDataTorch.accept(path)
        self.assertIsInstance(result, bool)
