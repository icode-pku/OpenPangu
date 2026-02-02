import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import numpy as np

from components.debug.compare.msquickcmp.caffe_model.caffe_dump_data import CaffeDumpData


class TestCaffeDumpData(unittest.TestCase):
    def setUp(self):
        self.arguments = MagicMock()
        self.arguments.input_shape = "data:1,3,224,224"
        self.arguments.model_path = "/path/to/model.prototxt"
        self.arguments.weight_path = "/path/to/weights.caffemodel"
        self.arguments.out_path = "/tmp/output"
        self.arguments.input_path = None

    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.utils.logger")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.utils.create_directory")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.utils.parse_input_shape", return_value=[(1, 3, 224, 224)])
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.CaffeDumpData._check_path_exists")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.CaffeDumpData._init_model")
    def test_init(self, mock_init_model, mock_check_path, mock_parse_input, mock_create_dir, mock_logger):
        mock_model = MagicMock()
        mock_init_model.return_value = mock_model

        obj = CaffeDumpData(self.arguments)

        mock_check_path.assert_any_call(self.arguments.model_path, extentions=[".prototxt"])
        mock_check_path.assert_any_call(self.arguments.weight_path, extentions=[".caffemodel", ".bin"])
        mock_create_dir.assert_any_call(os.path.join(obj.output_path, "input"))
        mock_create_dir.assert_any_call(os.path.join(obj.output_path, "dump_data", "caffe"))

        self.assertEqual(obj.model, mock_model)
        self.assertEqual(obj.input_data_path, None)

    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.CaffeDumpData._init_tensors_info")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.utils.logger")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.CaffeDumpData._generate_random_input_data")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.os.listdir", return_value=[])
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.CaffeDumpData._check_path_exists")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.CaffeDumpData._init_model")
    def test_generate_inputs_data_random(
        self, mock_init_model, mock_check_path, mock_listdir, mock_gen_input, mock_logger, mock_init_info
    ):
        mock_model = MagicMock()
        mock_model.inputs = ["data"]
        mock_init_model.return_value = mock_model
        mock_init_info.return_value = (["data"], [(1, 3, 224, 224)], ["float32"])
        mock_gen_input.return_value = {"data": np.zeros((1, 3, 224, 224))}

        obj = CaffeDumpData(self.arguments)
        obj.generate_inputs_data()

        self.assertIn("data", obj.inputs_map)

    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.CaffeDumpData._check_path_exists")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.CaffeDumpData._init_tensors_info")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.os.listdir")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.CaffeDumpData._read_input_data")
    def test_generate_inputs_data_from_file(self, mock_read_data, mock_listdir, mock_init_info, mock_check_path):
        mock_model = MagicMock()
        mock_model.inputs = ["data"]
        mock_init_info.return_value = (["data"], [(1, 3, 224, 224)], ["float32"])
        mock_listdir.return_value = ["input_0.bin", "input_1.bin"]
        mock_read_data.return_value = {"data": np.ones((1, 3, 224, 224))}

        with patch.object(CaffeDumpData, "_init_model", return_value=mock_model):
            obj = CaffeDumpData(self.arguments)
            obj.generate_inputs_data()

        mock_read_data.assert_called_once()

    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.CaffeDumpData._check_path_exists")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.CaffeDumpData._save_dump_data")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.CaffeDumpData._run_model")
    def test_generate_dump_data(self, mock_run_model, mock_save_dump, mock_check_path):
        mock_model = MagicMock()
        with patch.object(CaffeDumpData, "_init_model", return_value=mock_model):
            obj = CaffeDumpData(self.arguments)
            obj.inputs_map = {"data": np.zeros((1, 3, 224, 224))}
            dump_path = obj.generate_dump_data()
            self.assertEqual(dump_path, obj.dump_data_dir)

    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.CaffeDumpData._check_path_exists")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.np.save")
    @patch("components.debug.compare.msquickcmp.caffe_model.caffe_dump_data.utils.logger")
    def test_save_dump_data(self, mock_logger, mock_np_save, mock_check_path):
        mock_model = MagicMock()
        mock_model.outputs = ["output"]
        mock_model.top_names = {"conv1": ["output"]}
        mock_model.blobs = {
            "output": MagicMock(data=np.zeros((1, 3, 224, 224)))
        }

        with patch.object(CaffeDumpData, "_init_model", return_value=mock_model):
            obj = CaffeDumpData(self.arguments)
            obj._save_dump_data(mock_model)
            mock_np_save.assert_called_once()
