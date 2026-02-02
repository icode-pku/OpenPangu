import unittest
import os, sys
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestOpCheckerStartTest(unittest.TestCase):
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
        from msit_opcheck.opchecker import OpChecker
        self.OpChecker = OpChecker

    def tearDown(self):
        for mod in self.modules_to_mock:
            if mod in self.backup_modules:
                sys.modules[mod] = self.backup_modules[mod]
            else:
                del sys.modules[mod] 
        for mod in list(sys.modules.keys()):
            if mod.startswith("msit_opcheck.opchecker"):
                del sys.modules[mod]

    def test_check_input_path_argument(self):
        mock_args = MagicMock()
        mock_args.input = str("input_path")
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "input"))
            os.makedirs(os.path.join(tmpdir, "model"))
            os.makedirs(os.path.join(tmpdir, "dump_data", "npu", "123456", "789012"))
            with open(os.path.join(tmpdir, "model", "ge_proto_123.json"), "w") as f:
                f.write("{}")
            args = type('Args', (), {'input': tmpdir, 'output': None})
            checker = self.OpChecker(args)
            checker.check_input_path_argument(tmpdir)
            assert checker.ge_json_path is not None
            assert checker.origin_dump_path is not None
            assert "123456" in checker.origin_dump_path
            assert "789012" in checker.origin_dump_path

    @patch("msit_opcheck.opchecker.check_write_directory")
    @patch("os.makedirs")
    def test_init_output_file_path(self, mock_makedirs, mock_check_write_directory):
        mock_args = MagicMock()
        mock_args.output = str("output_path")
        checker = self.OpChecker(mock_args)
        checker.init_output_file_path()
        mock_makedirs.assert_called_once()

    @patch("components.utils.file_open_check.FileStat.is_basically_legal", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_update_dump_data_path(self, mock_path_isfile, mock_is_basically_legal):
        mock_args = MagicMock()
        checker = self.OpChecker(mock_args)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "123456", "789012"))
            checker.update_dump_data_path(tmpdir)
            self.assertEqual(checker.dump_data_path, os.path.join(tmpdir, "123456", "789012"))

    def test_clear_tmp_file(self):
        mock_args = MagicMock()
        checker = self.OpChecker(mock_args)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "tmp"))
            checker.npy_path = os.path.join(tmpdir, "tmp")
            test_file = Path(checker.npy_path) / "test.txt"
            test_file.write_text("test content")
            checker.clear_tmp_file(remove_dir=False)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "tmp")))
            checker.clear_tmp_file(remove_dir=True)
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "tmp")))

    def test_bind_op_info_to_case_info(self):
        mock_args = MagicMock()
        checker = self.OpChecker(mock_args)
        npy_files = [
            "prefix.op1.input.0.npy",
            "prefix.op1.input.1.npy",
            "prefix.op1.output.0.npy",
            "prefix.op2.input.0.npy",
            "prefix.op2.output.0.npy"
        ]
        op_info_dict = {
            "op1": MagicMock(),
            "op2": MagicMock()
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            for file in npy_files:
                Path(tmpdir).joinpath(file).touch()
            checker.bind_op_info_to_case_info(tmpdir, op_info_dict)
            for op_name in ["op1", "op2"]:
                assert op_name in op_info_dict
                op_info_dict[op_name].update_op_type.assert_called_once()

    @patch('msit_opcheck.opchecker.get_ge_graph_name')
    @patch('msit_opcheck.opchecker.convert_ge_dump_file_to_npy')
    @patch('msit_opcheck.opchecker.get_all_opinfo')
    @patch('msit_opcheck.opchecker.CaseManager')
    def test_start_test(self, mock_case_manager, mock_get_opinfo, mock_convert, mock_get_graph_name):
        mock_get_graph_name.return_value = ["graph_name"]
        mock_get_opinfo.return_value = {"op1": {"op_type": "Conv2D", "op_name": "conv1"}}
        mock_args = MagicMock()
        checker = self.OpChecker(mock_args)
        checker.ge_json_path = str("ge_graph.json")
        checker.origin_dump_path = str("self.test_input_dir")
        checker.cases_info = {"op1": {"excuted_information": "Success", "fail_reason": "unknown"}}
        mock_case_manager.return_value.add_case.return_value = (True, "")
        with patch("msit_opcheck.opchecker.OpChecker.check_input_path_argument"), \
             patch("msit_opcheck.opchecker.OpChecker.init_output_file_path"), \
             patch("msit_opcheck.opchecker.OpChecker.update_dump_data_path"), \
             patch("msit_opcheck.opchecker.OpChecker.clear_tmp_file"), \
             patch("msit_opcheck.opchecker.OpChecker.bind_op_info_to_case_info"), \
             patch("msit_opcheck.opchecker.check_input_path_legality"):
            checker.start_test()
        mock_convert.assert_called_once()
        mock_case_manager.return_value.add_case.assert_called()
        mock_case_manager.return_value.excute_cases.assert_called_once_with(1, "info")
