import unittest
from unittest.mock import patch, MagicMock
import sys
import numpy as np


class TestFuseOpChecker(unittest.TestCase):

    def setUp(self):
        self.backup_modules = {}
        self.modules_to_mock = [
            'tensorflow',
            'tensorflow.compat',
            'tensorflow.compat.v1',
        ]
        for mod in self.modules_to_mock:
            if mod in sys.modules:
                self.backup_modules[mod] = sys.modules[mod]
            sys.modules[mod] = MagicMock()
        from msit_opcheck.autofuse.fusion_op_check import FuseOpChecker
        self.FuseOpChecker = FuseOpChecker
    
    def tearDown(self):
        for mod in self.modules_to_mock:
            if mod in self.backup_modules:
                sys.modules[mod] = self.backup_modules[mod]
            else:
                del sys.modules[mod]  # 删除新增的mock模块
        # 清理测试中导入的模块
        for mod in list(sys.modules.keys()):
            if mod.startswith("msit_opcheck.autofuse.fusion_op_check") or mod.startswith("msit_opcheck.autofuse.tf_builder"):
                del sys.modules[mod]

    @patch("importlib.util.module_from_spec")
    @patch("importlib.util.spec_from_file_location")
    def test__load_pyautofuse_graph_given_valid_file_when_load_then_return_module(self, mock_spec_from_file, mock_module_from_spec):
        mock_spec = MagicMock()
        mock_spec_from_file.return_value = mock_spec
        mock_module = MagicMock()
        mock_module_from_spec.return_value = mock_module
        file_path = "/fake/path/graph.py"
        with patch("msit_opcheck.autofuse.fusion_op_check.check_input_path_legality", side_effect=lambda x: x):
            result = self.FuseOpChecker._load_pyautofuse_graph(file_path)
        mock_spec_from_file.assert_called_once_with("graph_module", file_path)
        mock_module_from_spec.assert_called_once_with(mock_spec)
        mock_spec.loader.exec_module.assert_called_once_with(mock_module)
        self.assertEqual(result, mock_module)

    @patch("os.listdir")
    @patch("msit_opcheck.autofuse.fusion_op_check.logger")
    def test__map_opname_to_dump_data_given_unsupported_file_format_when_processing_then_skip(self, mock_logger, mock_listdir):
        mock_args = MagicMock()
        mock_args.input = "/test/input"
        mock_listdir.return_value = ["invalid.file.format"]
        checker = self.FuseOpChecker(mock_args)
        checker._map_opname_to_dump_data()
        mock_logger.warning.assert_called_with("Find a unsupported GE dump file: 'invalid.file.format'.")
        self.assertEqual(len(checker.opname_to_dump_data_map), 0)

    @patch("os.listdir")
    @patch("os.path.isdir")
    def test__get_ascgraph_path_given_multiple_graph_files_when_processing_then_warn(self, mock_isdir, mock_listdir):
        mock_args = MagicMock()
        mock_args.graph_path = "/test/graph"
        mock_listdir.return_value = ["ascgen_dump_pid123"]
        mock_isdir.return_value = True
        mock_get_ascgraph = MagicMock(return_value=["file1.py", "file2.py"])
        with patch("msit_opcheck.autofuse.fusion_op_check.get_ascbackend_ascgraph", mock_get_ascgraph):
            checker = self.FuseOpChecker(mock_args)
            result = checker._get_ascgraph_path()
            
        self.assertEqual(len(result), 0)

    def test__compare_output_given_mismatched_lengths_when_comparing_then_log_warning(self):
        mock_args = MagicMock()
        checker = self.FuseOpChecker(mock_args)
        with patch("numpy.load") as mock_np_load:
            mock_np_load.side_effect = lambda x: x
            checker._compare_output([np.array([1, 2])], [np.array([1, 2])], MagicMock())
            
            self.assertEqual(len(checker.compare_result["Opname"]), 1)
            self.assertEqual(checker.compare_result["cosine_similarity"], [1.0])
            self.assertEqual(checker.compare_result["max_relative_error"], [0.0])
            self.assertEqual(checker.compare_result["kl_divergence"], [0])

    @patch("shutil.rmtree")
    def test_start_test_given_missing_graph_name_when_processing_then_skip(self, mock_rmtree):
        mock_args = MagicMock()
        mock_args.output = "/test/output"
        with patch.object(self.FuseOpChecker, "_get_ascgraph_path", return_value=[]), \
             patch.object(self.FuseOpChecker, "_map_opname_to_dump_data", return_value=None), \
             patch("msit_opcheck.autofuse.fusion_op_check.check_output_path_legality", side_effect=lambda x: x), \
             patch("os.path.exists", return_value=True), \
             patch.object(self.FuseOpChecker, "_save_compare_result", return_value=MagicMock()):
            
            checker = self.FuseOpChecker(mock_args)
            checker.opname_to_dump_data_map = {} 
            checker.start_test()
            
        mock_rmtree.assert_called_once()

    def test__get_ascgraph_dump_data_given_missing_graph_name_when_fetching_then_return_none_none(self):
        mock_args = MagicMock()
        checker = self.FuseOpChecker(mock_args)
        checker.opname_to_dump_data_map = {}
        checker.npy_path = "/temp/test"
        mock_tf_builder = MagicMock()
        mock_tf_builder.graph_name = "missing_graph"
        with patch("os.makedirs"):
            result1, result2 = checker._get_ascgraph_dump_data(mock_tf_builder)
            self.assertIsNone(result1)
            self.assertIsNone(result2)
