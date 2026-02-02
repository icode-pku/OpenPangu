# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

"""
Function:
This class is used to generate GUP dump data of the Caffe model.
"""
import os
import re

import numpy as np

from msquickcmp.common.dump_data import DumpData
from msquickcmp.common import utils


class CaffeDumpData(DumpData):
    """
    This class is used to generate GUP dump data of the Caffe model.
    """

    def __init__(self, arguments):
        super().__init__()
        input_shapes = utils.parse_input_shape(arguments.input_shape)
        if len(input_shapes) > 0:
            utils.logger.warning(
                "input_shapes provided, but currently dynamic input shape not supported in caffe, ignore."
            )
        
        self._check_path_exists(arguments.model_path, extentions=[".prototxt"])
        self._check_path_exists(arguments.weight_path, extentions=[".caffemodel", ".bin"])

        self.output_path = os.path.realpath(arguments.out_path)
        self.model_path = os.path.realpath(arguments.model_path)
        self.weight_path = os.path.realpath(arguments.weight_path)
        if arguments.input_path:
            self.input_data_path = [os.path.realpath(cur) for cur in arguments.input_path.split(",")]
        else:
            self.input_data_path = None

        self.input_data_save_dir = os.path.join(self.output_path, "input")
        self.dump_data_dir = os.path.join(self.output_path, "dump_data", "caffe")

        utils.create_directory(self.input_data_save_dir)
        utils.create_directory(self.dump_data_dir)

        self.model = self._init_model()

        self.inputs_map = {}
        self.net_output = {}

    @staticmethod
    def _init_tensors_info(model, tensor_names):
        names, shapes, dtypes = [], [], []
        for name in tensor_names:
            names.append(name)
            data = model.blobs[name].data
            shapes.append(data.shape)
            dtypes.append(data.dtype.name)
        return names, shapes, dtypes

    @staticmethod
    def _run_model(model, inputs_map):
        for input_name, input_data in inputs_map.items():
            np.copyto(model.blobs[input_name].data, input_data)
        return model.forward()  # {"output_name": output_numpy_data}

    def generate_inputs_data(self, npu_dump_data_path=None, use_aipp=False):
        input_names, input_shapes, input_dtypes = self._init_tensors_info(self.model, self.model.inputs)

        input_info = [
            {"name": name, "shape": shape, "type": dtype}
            for name, shape, dtype in zip(input_names, input_shapes, input_dtypes)
        ]
        utils.logger.info("Caffe input info: \n{}\n".format(input_info))

        if self.input_data_path:
            self._check_input_data_path(self.input_data_path, input_shapes)
            self.inputs_map = self._read_input_data(self.input_data_path, input_names, input_shapes, input_dtypes)
        elif os.listdir(self.input_data_save_dir):
            input_bin_files = os.listdir(self.input_data_save_dir)
            input_bin_files.sort(key=lambda file: int((re.findall("\\d+", file))[0]))
            bin_file_path_array = [os.path.join(self.input_data_save_dir, item) for item in input_bin_files]
            self.inputs_map = self._read_input_data(bin_file_path_array, input_names, input_shapes, input_dtypes)
        else:
            self.inputs_map = self._generate_random_input_data(
                self.input_data_save_dir, input_names, input_shapes, input_dtypes
            )

    def generate_dump_data(self, npu_dump_path=None, om_parser=None):
        """
        Function description:
            generate caffe model dump data
        Parameter:
            none
        Return Value:
            caffe model dump data directory
        Exception Description:
            none
        """
        self._run_model(self.model, self.inputs_map)
        self._save_dump_data(self.model)
        return self.dump_data_dir

    def _init_model(self):
        import caffe

        model = caffe.Net(self.model_path, self.weight_path, caffe.TEST)
        return model

    def _save_dump_data(self, model):
        output_names = model.outputs
        for layer_name, blob_names in model.top_names.items():
            for blob_id, blob_name in enumerate(blob_names):
                file_name = self._generate_dump_data_file_name(layer_name, blob_id)
                file_path = os.path.join(self.dump_data_dir, file_name)
                np.save(file_path, model.blobs.get(blob_name, np.empty(0, dtype="float32")).data)
                utils.logger.info(f"The dump data of layer '{layer_name}' has been saved to '{file_path}'")

                if blob_name in output_names:
                    self.net_output[output_names.index(blob_name)] = file_path

        for name, file_path in self.net_output.items():
            utils.logger.info("net_output node is:{}, file path is {}".format(name, file_path))
        utils.logger.info("dump data success")
