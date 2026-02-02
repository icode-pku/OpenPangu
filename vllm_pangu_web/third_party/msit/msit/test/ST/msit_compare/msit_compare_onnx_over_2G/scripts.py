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

import logging
import os

logger = logging.getLogger(__name__)
LOG_LEVEL = os.environ.get("LOG_LEVEL", 'ERROR')

logging.basicConfig(
    level=LOG_LEVEL,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %I:%M:%S %p'
)

try:
    import torch
    import torchvision
except Exception as e:
    logger.error("Something wrong with `torch` or `torchvision`. Try reinstall if their versions are not compatible.")
    raise


class LargeNet(torch.nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        self.base_model = torchvision.models.resnet152()
        self.fc1 = torch.nn.Linear(1000, 2048)
        self.fc2 = torch.nn.Linear(2048, 4096)
        self.fc3 = torch.nn.Linear(4096, 4096 * 2)
        self.fc4 = torch.nn.Linear(4096 * 2, 4096 * 4)
        self.fc5 = torch.nn.Linear(4096 * 4, 4096 * 10)

    def forward(self, x):
        x = self.base_model(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))

        return x
    

if __name__ == "__main__":
    try:
        import onnx
    except Exception as e:
        logger.error("onnx error. Try reinstall onnx.")
        raise

    try:
        os.mkdir("temp_dir", 0o750)
    except Exception as e:
        logger.error("%s cannot be created. See if it exists or do not have permission.", "temp_dir")
        raise

    logger.debug("%s has being created successfully.", os.path.abspath("temp_dir"))

    model = LargeNet()

    try:
        torch.onnx.export(model, torch.randn(1, 3, 2, 4), "temp_dir/large_model.onnx")
    except Exception as e:
        logger.error("Error occured when trying to save the onnx model.")
    else:
        logger.debug("%s has being saved successfully", os.path.abspath("temp_dir/large_model.onnx"))
