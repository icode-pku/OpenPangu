# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd. All rights reserved.
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
from queue import Queue
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

from components.debug.surgeon.auto_optimizer.inference_engine.inference.onnx_inference import ONNXInference

 
# Mock classes and methods
class MockONNXInference(ONNXInference):
    def __init__(self):
        super().__init__()
        self.model = "test_model.onnx"
 
    def _get_params(self, cfg):
        return self.model
 

@pytest.fixture(scope="module")
def onnx_inference_instance():
    return MockONNXInference()
 

@pytest.fixture(scope="module", autouse=True)
def mock_external_dependencies():
    with patch("onnxruntime.InferenceSession") as mock_session:
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
 
        mock_session_instance.get_inputs.return_value = [
            {"name": "input_1", "shape": None, "type": None, "dtype": np.float32},
        ]
        mock_session_instance.get_outputs.return_value = [
            {"name": "output_1", "shape": None, "type": None, "dtype": np.float32},
        ]
        mock_session_instance.run.return_value = [np.array([1.0, 2.0, 3.0])]
 
        with patch("components.debug.common.logger") as mock_logger:
            yield
 

def test_onnx_inference_given_invalid_model_when_get_params_then_raise_error(onnx_inference_instance):
    onnx_inference_instance._get_params = MagicMock(side_effect=Exception("Model param error"))
 
    with pytest.raises(RuntimeError, match="model gets params error"):
        onnx_inference_instance(1, {}, Queue(), Queue())
 

if __name__ == "__main__":
    unittest.main()
