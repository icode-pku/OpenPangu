# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import os
import re
from abc import ABC

from auto_optimizer.inference_engine.datasets.dataset_base import DatasetBase
from auto_optimizer.inference_engine.data_process_factory import DatasetFactory
from components.debug.common import logger
from components.utils.file_open_check import is_legal_args_path_string
from components.utils.file_open_check import ms_open
from components.utils.check.rule import Rule
from components.utils.constants import TENSOR_MAX_SIZE


@DatasetFactory.register("imagenet")
class ImageNetDataset(DatasetBase, ABC):
    def __call__(self, batch_size, cfg, in_queue, out_queue):
        """
        和基类的参数顺序和个数需要一致
        """
        logger.debug("dataset start")
        dataset_path, label_path = super()._get_params(cfg)

        data = []
        labels = []
        try:
            Rule.input_file().check(label_path, will_raise=True)
            with ms_open(label_path, 'r', max_size=TENSOR_MAX_SIZE) as f:
                for label_file in f:
                    image_name, label = re.split(r"\s+", label_file.strip())
                    file_path = os.path.join(dataset_path, image_name)
                    if not is_legal_args_path_string(file_path):
                        logger.warning("The file path of %r is not legal, skip this image and label", image_name)
                        continue

                    labels.append(label)
                    data.append(file_path)

                    if len(data) == batch_size:
                        out_queue.put([labels, data])
                        labels.clear()
                        data.clear()

                while data and len(data) < batch_size:
                    labels.append(labels[0])  # 数据补齐
                    data.append(data[0])
                    out_queue.put([labels, data])

        except Exception as err:
            logger.error("pre_process failed error={}".format(err))

        logger.debug("dataset end")
