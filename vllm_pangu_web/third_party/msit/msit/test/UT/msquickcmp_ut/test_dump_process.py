import unittest
from unittest.mock import patch, MagicMock
from components.debug.compare.msquickcmp.dump.dump_process import _generate_golden_data_model, dump_process, \
    dump_data, npu_dump_process, cpu_dump_process, check_and_dump


class TestGenerateGoldenDataModel(unittest.TestCase):
    @patch("components.debug.compare.msquickcmp.dump.dump_process.is_saved_model_valid", return_value=True)
    @patch("components.debug.compare.msquickcmp.tf.tf_save_model_dump_data.TfSaveModelDumpData", autospec=True)
    def test_saved_model_valid(self, mock_tf_model, mock_is_valid):
        args = MagicMock()
        args.model_path = "some/saved_model"
        result = _generate_golden_data_model(args, "npu_path")
        mock_tf_model.assert_called_once_with(args, args.model_path)
        self.assertEqual(result, mock_tf_model.return_value)

    @patch("components.debug.compare.msquickcmp.dump.dump_process.utils.get_model_name_and_extension", return_value=("model", ".onnx"))
    @patch("components.debug.compare.msquickcmp.onnx_model.onnx_dump_data.OnnxDumpData", autospec=True)
    @patch("components.debug.compare.msquickcmp.dump.dump_process.is_saved_model_valid", return_value=False)
    def test_onnx_model(self, mock_is_valid, mock_onnx, mock_get_name):
        args = MagicMock()
        args.model_path = "model.onnx"
        result = _generate_golden_data_model(args, "npu_path")
        mock_onnx.assert_called_once_with(args, "npu_path")
        self.assertEqual(result, mock_onnx.return_value)


class TestDumpProcess(unittest.TestCase):
    @patch("components.debug.compare.msquickcmp.dump.dump_process.convert_npy_to_bin", return_value="converted_input")
    @patch("components.debug.compare.msquickcmp.dump.dump_process.check_and_dump")
    @patch("os.path.realpath", side_effect=lambda x: f"/abs/{x}")
    def test_dump_process(self, mock_realpath, mock_check_and_dump, mock_convert):
        args = MagicMock()
        args.model_path = "model"
        args.weight_path = "weights"
        args.cann_path = "cann"
        args.input_path = "input"
        args.fusion_switch_file = "fusion"
        dump_process(args, use_cli=True)
        self.assertEqual(args.model_path, "/abs/model")
        self.assertEqual(args.input_path, "converted_input")
        mock_check_and_dump.assert_called_once()


class TestDumpData(unittest.TestCase):
    @patch("components.debug.compare.msquickcmp.dump.dump_process.get_shape_to_directory_name", return_value="shape_dir")
    @patch("components.debug.compare.msquickcmp.dump.dump_process.npu_dump_process")
    def test_npu_branch(self, mock_npu, mock_shape_dir):
        args = MagicMock(device_pattern="npu")
        dump_data(args, "1,2,3", "base", use_cli=False)
        mock_npu.assert_called_once()

    @patch("components.debug.compare.msquickcmp.dump.dump_process.get_shape_to_directory_name", return_value="shape_dir")
    @patch("components.debug.compare.msquickcmp.dump.dump_process.cpu_dump_process")
    def test_cpu_branch(self, mock_cpu, mock_shape_dir):
        args = MagicMock(device_pattern="cpu")
        dump_data(args, "1,2,3", "base", use_cli=False)
        mock_cpu.assert_called_once()

    def test_invalid_device(self):
        args = MagicMock(device_pattern="other")
        with self.assertRaises(ValueError):
            dump_data(args, None, "path", use_cli=False)


class TestNpuDumpProcess(unittest.TestCase):
    @patch("components.debug.compare.msquickcmp.dump.dump_process._generate_model_adapter")
    def test_npu_dump_process(self, mock_gen_adapter):
        dumper = MagicMock()
        mock_gen_adapter.return_value = dumper
        args = MagicMock()
        npu_dump_process(args, use_cli=True)
        dumper.generate_inputs_data.assert_called_once_with(use_aipp=False)
        dumper.generate_dump_data.assert_called_once_with(use_cli=True)

class TestCpuDumpProcess(unittest.TestCase):
    @patch("components.debug.compare.msquickcmp.dump.dump_process._generate_golden_data_model")
    @patch("components.debug.compare.msquickcmp.dump.dump_process.is_saved_model_valid", return_value=True)
    def test_saved_model_with_tf_json(self, mock_valid, mock_gen_model):
        dumper = MagicMock()
        mock_gen_model.return_value = dumper
        args = MagicMock(model_path="path", tf_json_path="file.json")
        cpu_dump_process(args)
        dumper.generate_inputs_data_for_dump.assert_called_once()
        dumper.generate_dump_data.assert_called_once_with("file.json")

    @patch("components.debug.compare.msquickcmp.dump.dump_process._generate_golden_data_model")
    @patch("components.debug.compare.msquickcmp.dump.dump_process.is_saved_model_valid", return_value=True)
    def test_saved_model_without_tf_json(self, mock_valid, mock_gen_model):
        dumper = MagicMock()
        mock_gen_model.return_value = dumper
        args = MagicMock(model_path="path", tf_json_path=None)
        with self.assertRaises(ValueError):
            cpu_dump_process(args)

    @patch("components.debug.compare.msquickcmp.dump.dump_process.utils.get_model_name_and_extension", return_value=("model", ".pb"))
    @patch("components.debug.compare.msquickcmp.dump.dump_process._generate_golden_data_model")
    @patch("components.debug.compare.msquickcmp.dump.dump_process.is_saved_model_valid", return_value=False)
    def test_other_model_type(self, mock_valid, mock_gen_model, mock_get_ext):
        dumper = MagicMock()
        mock_gen_model.return_value = dumper
        args = MagicMock(model_path="path")
        cpu_dump_process(args)
        dumper.generate_inputs_data.assert_called_once_with(npu_dump_data_path=None, use_aipp=False)
        dumper.generate_dump_data.assert_called_once()


class TestCheckAndDump(unittest.TestCase):
    @patch("components.debug.compare.msquickcmp.dump.dump_process.dump_data")
    @patch("components.debug.compare.msquickcmp.dump.dump_process.utils.parse_dym_shape_range", return_value=["shape1", "shape2"])
    @patch("components.debug.compare.msquickcmp.dump.dump_process.utils.check_file_or_directory_path")
    @patch("components.debug.compare.msquickcmp.dump.dump_process.utils.check_device_param_valid")
    @patch("components.debug.compare.msquickcmp.dump.dump_process.is_saved_model_valid", return_value=True)
    @patch("os.path.realpath", side_effect=lambda p: f"/abs/{p}")
    @patch("time.strftime", return_value="20250610123456")
    def test_check_and_dump_with_dym_shapes(self, mock_time, mock_realpath, mock_valid, mock_check_device,
                                            mock_check_file, mock_parse_shape, mock_dump):
        args = MagicMock(
            model_path="model",
            weight_path="weight",
            fusion_switch_file="fusion",
            device="npu",
            out_path="out",
            dym_shape_range="shape_range"
        )
        check_and_dump(args, use_cli=True)
        self.assertTrue(mock_dump.call_count == 2)

    @patch("components.debug.compare.msquickcmp.dump.dump_process.dump_data")
    @patch("components.debug.compare.msquickcmp.dump.dump_process.utils.check_file_or_directory_path")
    @patch("components.debug.compare.msquickcmp.dump.dump_process.utils.check_device_param_valid")
    @patch("components.debug.compare.msquickcmp.dump.dump_process.is_saved_model_valid", return_value=True)
    @patch("os.path.realpath", side_effect=lambda p: f"/abs/{p}")
    @patch("time.strftime", return_value="20250610123456")
    def test_check_and_dump_no_dym_shape(self, mock_time, mock_realpath, mock_valid, mock_check_device,
                                         mock_check_file, mock_dump):
        args = MagicMock(
            model_path="model",
            weight_path=None,
            fusion_switch_file=None,
            device="cpu",
            out_path="out",
            dym_shape_range=None
        )
        check_and_dump(args, use_cli=False)
        mock_dump.assert_called_once()
