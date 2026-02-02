import sys
import unittest
from unittest.mock import patch, MagicMock
from argparse import Namespace
from components.debug.compare.msquickcmp.__main__ import CompareCommand, check_normal_dump_param, \
    get_compare_cmd_ins, get_dump_cmd_ins


class TestCompareCommand(unittest.TestCase):

    @patch('components.debug.compare.msquickcmp.__main__.logger')
    def test_handle_no_golden_model_and_no_golden_path(self, mock_logger):
        cmd = CompareCommand(name="compare", help_info="compare data")
        cmd.parser = MagicMock()
        args = Namespace(golden_model=None, golden_path=None, ops_json=None)

        cmd.handle(args)

        mock_logger.error.assert_called_once()
        cmd.parser.print_help.assert_called_once()

    @patch('components.debug.compare.msquickcmp.__main__.check_ops_json_path', side_effect=lambda x: x)
    @patch('components.debug.compare.msquickcmp.__main__.check_om_path_legality', side_effect=lambda x: x)
    @patch('components.debug.compare.msquickcmp.__main__.os.path.exists')
    @patch('components.debug.compare.msquickcmp.__main__.check_input_data_path_legality', side_effect=lambda x: x)
    @patch('msquickcmp.mie_torch.mietorch_comp.MIETorchCompare')
    @patch('components.debug.compare.msquickcmp.__main__.os.path.join', side_effect=lambda *args: "/".join(args))
    def test_handle_ops_json_present(self, mock_join, mock_mietorch, mock_check_path, mock_exists, mock_mietorch_path,
                                     mock_ops_json_path):
        mock_exists.return_value = True
        args = Namespace(
            golden_model=None,
            golden_path='golden_path',
            my_path='my_path',
            ops_json='ops_json',
            out_path='out_path'
        )
        comparer_mock = MagicMock()
        mock_mietorch.return_value = comparer_mock
        cmd = CompareCommand(name="compare", help_info="compare data")
        cmd.parser = MagicMock()

        cmd.handle(args)

        mock_mietorch.assert_called_once_with('golden_path', 'my_path', 'ops_json', 'out_path')
        comparer_mock.compare.assert_called_once()

    @patch('components.debug.compare.msquickcmp.__main__.cmp_process')
    @patch('components.debug.compare.msquickcmp.__main__.CmpArgsAdapter')
    def test_handle_normal_compare_flow(self, mock_adapter, mock_process):
        args = Namespace(
            golden_model='model.pb',
            golden_path=None,
            om_model='model.om',
            weight_path=None,
            input_data_path='input.bin',
            cann_path='/path/to/cann',
            out_path='./output',
            input_shape='',
            device='0',
            output_size='',
            output_nodes='',
            advisor=False,
            dym_shape_range='',
            dump=True,
            bin2npy=False,
            custom_op='',
            locat=False,
            onnx_fusion_switch=True,
            single_op=False,
            fusion_switch_file=None,
            max_cmp_size=0,
            quant_fusion_rule_file='',
            saved_model_signature='serving_default',
            saved_model_tag_set='serve',
            my_path=None,
            ops_json=None
        )

        cmd = CompareCommand(name="compare", help_info="compare data")
        cmd.parser = MagicMock()

        cmd.handle(args)

        mock_adapter.assert_called_once()
        mock_process.assert_called_once()


class TestCheckNormalDumpParam(unittest.TestCase):

    def test_raise_if_opname_true(self):
        args = Namespace(opname=True, exec=False)
        with self.assertRaises(NotImplementedError) as context:
            check_normal_dump_param(args)
        self.assertIn('--operation-name', str(context.exception))

    def test_raise_if_exec_true(self):
        args = Namespace(opname=False, exec=True)
        with self.assertRaises(NotImplementedError) as context:
            check_normal_dump_param(args)
        self.assertIn('--exec', str(context.exception))

    def test_no_exception_when_both_false(self):
        args = Namespace(opname=False, exec=False)
        try:
            check_normal_dump_param(args)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")


class TestDumpCommand(unittest.TestCase):
    def setUp(self):
        from components.debug.compare.msquickcmp.__main__ import DumpCommand
        self.command = DumpCommand("dump", "help")

    @patch("components.debug.compare.msquickcmp.__main__.is_enough_disk_space_left", return_value=True)
    @patch("components.debug.compare.msquickcmp.__main__.filter_cmd", return_value=["bash", "run.sh", "script.py"])
    @patch("components.debug.compare.msquickcmp.__main__.subprocess.run")
    def test_handle_with_exec(self, mock_run, mock_filter_cmd, mock_disk_space):
        with patch.dict(sys.modules, {
            "components.debug.compare.msquickcmp.dump.mietorch.dump_config": MagicMock()
        }):
            from components.debug.compare.msquickcmp.dump.mietorch.dump_config import DumpConfig
            mock_dump_config = DumpConfig

            mock_args = MagicMock()
            mock_args.out_path = "./output"
            mock_args.exec = "bash run.sh script.py"
            mock_args.opname = "op_name"

            self.command.handle(mock_args)

            mock_disk_space.assert_called_once_with("./output")
            mock_dump_config.assert_called_once_with(dump_path="./output", api_list="op_name")
            mock_filter_cmd.assert_called_once()
            mock_run.assert_called_once_with(["bash", "run.sh", "script.py"], shell=False)
    
    @patch("components.debug.compare.msquickcmp.__main__.is_enough_disk_space_left", return_value=True)
    @patch("components.debug.compare.msquickcmp.__main__.check_normal_dump_param")
    @patch("components.debug.compare.msquickcmp.__main__.dump_process")
    @patch("components.debug.compare.msquickcmp.__main__.DumpArgsAdapter")
    def test_handle_with_model_and_device(self, mock_adapter_cls, mock_dump_process, mock_check_param, mock_disk_space):
        mock_args = MagicMock()
        mock_args.out_path = ""
        mock_args.exec = None
        mock_args.model_path = "model.onnx"
        mock_args.device_pattern = "cpu"
        mock_args.weight_path = "weight"
        mock_args.input_data_path = "input.bin"
        mock_args.cann_path = "/usr/local/cann"
        mock_args.input_shape = "input:1,224,224,3"
        mock_args.device = "0"
        mock_args.dym_shape_range = ""
        mock_args.onnx_fusion_switch = True
        mock_args.saved_model_signature = "serving_default"
        mock_args.saved_model_tag_set = "serve"
        mock_args.tf_json_path = None
        mock_args.fusion_switch_file = None

        mock_adapter = MagicMock()
        mock_adapter_cls.return_value = mock_adapter

        self.command.handle(mock_args)

        mock_disk_space.assert_called_once_with("./")
        mock_check_param.assert_called_once_with(mock_args)
        mock_adapter_cls.assert_called_once()
        mock_dump_process.assert_called_once_with(mock_adapter, True)

    @patch("components.debug.compare.msquickcmp.__main__.is_enough_disk_space_left", return_value=False)
    def test_handle_raises_if_disk_space_low(self, mock_disk_space):
        mock_args = MagicMock()
        mock_args.out_path = ""
        from components.debug.compare.msquickcmp.__main__ import DumpCommand
        self.command = DumpCommand("dump", "help")

        with self.assertRaises(OSError):
            self.command.handle(mock_args)

    def test_handle_raises_if_missing_model_and_device(self):
        mock_args = MagicMock()
        mock_args.out_path = "some_path"
        mock_args.exec = None
        mock_args.model_path = None
        mock_args.device_pattern = None

        with patch("components.debug.compare.msquickcmp.__main__.is_enough_disk_space_left", return_value=True):
            with self.assertRaises(NotImplementedError):
                self.command.handle(mock_args)


class TestCommandInstances(unittest.TestCase):

    @patch('components.debug.compare.msquickcmp.__main__.CompareCommand')
    def test_get_compare_cmd_ins(self, mock_compare_command):
        mock_instance = MagicMock()
        mock_compare_command.return_value = mock_instance

        result = get_compare_cmd_ins()
        mock_compare_command.assert_called_once_with("compare", "one-click network-wide accuracy analysis of golden models.")
        self.assertEqual(result, mock_instance)

    @patch('components.debug.compare.msquickcmp.__main__.DumpCommand')
    def test_get_dump_cmd_ins(self, mock_dump_command):
        mock_instance = MagicMock()
        mock_dump_command.return_value = mock_instance
        result = get_dump_cmd_ins()
        mock_dump_command.assert_called_once_with("dump", "one-click dump model ops inputs and outputs.")
        self.assertEqual(result, mock_instance)
