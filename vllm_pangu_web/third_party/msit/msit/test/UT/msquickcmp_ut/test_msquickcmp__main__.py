import unittest
from unittest.mock import patch, MagicMock, call
import argparse

from msquickcmp.__main__ import (
    CompareCommand,
    DumpCommand,
    get_compare_cmd_ins,
    get_dump_cmd_ins,
    check_normal_dump_param,
)

class TestCompareCommand(unittest.TestCase):
    def setUp(self):
        self.cmd = CompareCommand("compare", "help")
        self.parser = MagicMock()
        self.cmd.add_arguments(self.parser)

    def test_add_arguments(self):
        calls = [
            call.add_argument('-gm', '--golden-model', required=False, dest="golden_model", type=unittest.mock.ANY, help=unittest.mock.ANY),
            call.add_argument('-om', '--om-model', required=False, dest="om_model", type=unittest.mock.ANY, help=unittest.mock.ANY),
            call.add_argument('-w', '--weight', dest="weight_path", type=unittest.mock.ANY, help=unittest.mock.ANY),
            call.add_argument('-i', '--input', default='', dest="input_data_path", type=unittest.mock.ANY, help=unittest.mock.ANY),
        ]
        self.parser.assert_has_calls(calls, any_order=True)

    @patch("msquickcmp.__main__.logger")
    def test_handle_missing_golden_model_and_path(self, mock_logger):
        args = argparse.Namespace(
            golden_model=None, golden_path=None, ops_json=None
        )
        self.cmd.parser = MagicMock()
        self.cmd.handle(args)
        mock_logger.error.assert_called_with("The following args are required: -gm/--golden-model or -gp/--golden-path")
        self.cmd.parser.print_help.assert_called_once()

    @patch("msquickcmp.__main__.check_ops_json_path", side_effect=lambda x: x)
    @patch("msquickcmp.__main__.check_om_path_legality", side_effect=lambda x: x)
    @patch("msquickcmp.__main__.os.path.exists", return_value=True)
    @patch("msquickcmp.__main__.check_input_data_path_legality", side_effect=lambda x: x)
    @patch("msquickcmp.__main__.os.path.join", side_effect=lambda *a: "/".join(a))
    @patch("msquickcmp.mie_torch.mietorch_comp.MIETorchCompare")
    def test_handle_ops_json_mindie(self, mock_mietorch, mock_join, mock_check, mock_exists, mock_mietorch_path,
                                    mock_ops_json_path):
        args = argparse.Namespace(
            golden_model="gm", golden_path="gp", ops_json="/tmp/ops", my_path="mp", out_path="out"
        )
        self.cmd.parser = MagicMock()
        self.cmd.handle(args)
        mock_mietorch.assert_called_once_with("gp", "mp", "/tmp/ops", "out")
        mock_mietorch.return_value.compare.assert_called_once()

    @patch("msquickcmp.__main__.CmpArgsAdapter")
    @patch("msquickcmp.__main__.cmp_process")
    def test_handle_normal_path(self, mock_cmp_process, mock_adapter):
        args = argparse.Namespace(
            golden_model="gm", om_model="om", weight_path="w", input_data_path="i", cann_path="c", out_path="o",
            input_shape="is", device="d", output_size="os", output_nodes="on", advisor=False, dym_shape_range="dr",
            dump=True, bin2npy=False, custom_op="", locat=False, onnx_fusion_switch=True, single_op=False,
            fusion_switch_file=None, max_cmp_size=0, quant_fusion_rule_file="", saved_model_signature="sig",
            saved_model_tag_set="tag", my_path=None, golden_path="gp", ops_json=None
        )
        self.cmd.parser = MagicMock()
        self.cmd.handle(args)
        mock_adapter.assert_called_once()
        mock_cmp_process.assert_called_once_with(mock_adapter.return_value, True)

class TestDumpCommand(unittest.TestCase):
    def setUp(self):
        self.cmd = DumpCommand("dump", "help")
        self.parser = MagicMock()
        self.cmd.add_arguments(self.parser)

    def test_add_arguments(self):
        calls = [
            call.add_argument('-m', '--model', required=False, dest="model_path", type=unittest.mock.ANY, help=unittest.mock.ANY),
            call.add_argument('-w', '--weight', dest="weight_path", type=unittest.mock.ANY, help=unittest.mock.ANY),
            call.add_argument('-i', '--input', default='', dest="input_data_path", type=unittest.mock.ANY, help=unittest.mock.ANY),
        ]
        self.parser.assert_has_calls(calls, any_order=True)

    @patch("msquickcmp.__main__.is_enough_disk_space_left", return_value=False)
    def test_handle_not_enough_disk(self, mock_disk):
        args = argparse.Namespace(
            out_path="", exec=None, model_path="m", device_pattern="cpu", weight_path=None,
            input_data_path=None, cann_path=None, input_shape=None, device=None, dym_shape_range=None,
            onnx_fusion_switch=True, saved_model_signature=None, saved_model_tag_set=None, tf_json_path=None,
            fusion_switch_file=None, opname=None
        )
        with self.assertRaises(OSError):
            self.cmd.handle(args)

    @patch("msquickcmp.__main__.is_enough_disk_space_left", return_value=True)
    @patch("msquickcmp.__main__.filter_cmd", side_effect=lambda x: x)
    @patch("msquickcmp.__main__.subprocess.run")
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.DumpConfig")
    def test_handle_exec(self, mock_dumpconfig, mock_subproc, mock_filter, mock_disk):
        args = argparse.Namespace(
            out_path="out", exec="bash run.sh", opname="api", model_path=None, device_pattern=None,
            weight_path=None, input_data_path=None, cann_path=None, input_shape=None, device=None,
            dym_shape_range=None, onnx_fusion_switch=True, saved_model_signature=None, saved_model_tag_set=None,
            tf_json_path=None, fusion_switch_file=None
        )
        self.cmd.handle(args)
        mock_dumpconfig.assert_called_once_with(dump_path="out", api_list="api")
        mock_subproc.assert_called_once()
        mock_filter.assert_called_once()

    @patch("msquickcmp.__main__.is_enough_disk_space_left", return_value=True)
    def test_handle_missing_model_path_or_device_pattern(self, mock_disk):
        args = argparse.Namespace(
            out_path="out", exec=None, model_path=None, device_pattern=None, weight_path=None,
            input_data_path=None, cann_path=None, input_shape=None, device=None, dym_shape_range=None,
            onnx_fusion_switch=True, saved_model_signature=None, saved_model_tag_set=None, tf_json_path=None,
            fusion_switch_file=None, opname=None
        )
        with self.assertRaises(NotImplementedError):
            self.cmd.handle(args)

    @patch("msquickcmp.__main__.is_enough_disk_space_left", return_value=True)
    @patch("msquickcmp.__main__.dump_process")
    @patch("msquickcmp.__main__.DumpArgsAdapter")
    @patch("msquickcmp.__main__.check_normal_dump_param")
    def test_handle_normal(self, mock_check, mock_adapter, mock_dump_process, mock_disk):
        args = argparse.Namespace(
            out_path="out", exec=None, model_path="m", device_pattern="cpu", weight_path="w",
            input_data_path="i", cann_path="c", input_shape="is", device="d", dym_shape_range="dr",
            onnx_fusion_switch=True, saved_model_signature="sig", saved_model_tag_set="tag", tf_json_path="tfj",
            fusion_switch_file="fs", opname=None
        )
        self.cmd.handle(args)
        mock_check.assert_called_once_with(args)
        mock_adapter.assert_called_once()
        mock_dump_process.assert_called_once_with(mock_adapter.return_value, True)

class TestCheckNormalDumpParam(unittest.TestCase):
    def test_opname_raises(self):
        args = argparse.Namespace(opname="op", exec=None)
        with self.assertRaises(NotImplementedError):
            check_normal_dump_param(args)

    def test_exec_raises(self):
        args = argparse.Namespace(opname=None, exec="run")
        with self.assertRaises(NotImplementedError):
            check_normal_dump_param(args)

class TestGetCmdIns(unittest.TestCase):
    def test_get_compare_cmd_ins(self):
        cmd = get_compare_cmd_ins()
        self.assertIsInstance(cmd, CompareCommand)
        self.assertEqual(cmd.name, "compare")
        self.assertIn("accuracy analysis", cmd.help_info)

    def test_get_dump_cmd_ins(self):
        cmd = get_dump_cmd_ins()
        self.assertIsInstance(cmd, DumpCommand)
        self.assertEqual(cmd.name, "dump")
        self.assertIn("dump model ops", cmd.help_info)

if __name__ == "__main__":
    unittest.main()