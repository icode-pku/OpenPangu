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

from abc import ABC

import onnxruntime as rt
import numpy as np

from auto_optimizer.inference_engine.inference.inference_base import InferenceBase
from auto_optimizer.inference_engine.data_process_factory import InferenceFactory
from components.debug.common import logger
from components.utils.check.rule import Rule


@InferenceFactory.register("onnx")
class ONNXInference(InferenceBase, ABC):

    def __call__(self, loop, cfg, in_queue, out_queue):
        """
        和基类的参数顺序和个数需要一致
        """
        logger.debug("inference start")
        try:
            model = super()._get_params(cfg)
        except Exception as err:
            logger.error("inference failed error={}".format(err))
            raise RuntimeError("model gets params error") from err
        try:
            session, input_name, output_name = self._session_init(model)
        except Exception as err:
            logger.error("inference failed error={}".format(err))
            raise RuntimeError("inference failed error") from err
        for _ in range(loop):
            data = in_queue.get()
            if len(data) < 2:  # include file_name and data
                raise RuntimeError("input params error len={}".format(len(data)))
            out_data = self._session_run(session, input_name, [data[1]])

            out_queue.put([data[0], out_data])

        logger.debug("inference end")

    def _session_init(self, model):
        if Rule.input_file().check(model, will_raise=True):
            session = rt.InferenceSession(model)
        input_name = []
        for n in session.get_inputs():
            input_name.append(n.name)
        output_name = []
        for n in session.get_outputs():
            output_name.append(n.name)

        return session, input_name, output_name

    def _session_run(self, session, input_name, input_data):
        res_buff = []

        res = session.run(None, {input_name[i]: input_data[i] for i in range(len(input_name))})
        for _, x in enumerate(res):
            out = np.array(x)
            res_buff.append(out)

        return res_buff
