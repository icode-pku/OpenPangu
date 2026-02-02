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

import re
import sys
from abc import ABC
import subprocess

import numpy as np

from auto_optimizer.inference_engine.inference.inference_base import InferenceBase
from auto_optimizer.inference_engine.data_process_factory import InferenceFactory
from components.debug.common import logger
from components.utils.util import filter_cmd
from components.utils.file_utils import check_file_or_directory_path

try:
    from ais_bench.infer.interface import InferSession
    import aclruntime
except ImportError:
    logger.warning('Failed to import benchmark module, please install extra [inference] feature.')

if 'aclruntime' in sys.modules:
    tensor_type_to_numpy_type = {
        aclruntime.dtype.int8: np.int8,
        aclruntime.dtype.uint8: np.uint8,
        aclruntime.dtype.int16: np.int16,
        aclruntime.dtype.uint16: np.uint16,
        aclruntime.dtype.int32: np.int32,
        aclruntime.dtype.uint32: np.uint32,
        aclruntime.dtype.int64: np.int64,
        aclruntime.dtype.uint64: np.uint64,
        aclruntime.dtype.float16: np.float16,
        aclruntime.dtype.float32: np.float32,
        aclruntime.dtype.double: np.double,
        aclruntime.dtype.bool: np.bool_,
    }



@InferenceFactory.register("acl")
class AclInference(InferenceBase, ABC):

    def __init__(self):
        # support msame and pyacl
        super().__init__()
        self.tool = 'pyacl'

    def __call__(self, loop, cfg, in_queue, out_queue):
        logger.debug("inference start")
        check_file_or_directory_path(cfg['model'], is_strict=True)

        if self.tool == 'msame':
            msame_cmd = [
                cfg['bin_path'],
                '--model={}'.format(cfg['model']),
                '--output={}'.format(cfg['output']),
                '--outfmt={}'.format(cfg['outfmt']),
                '--loop={}'.format(loop),
            ]
            msame_cmd = filter_cmd(msame_cmd)
            out = subprocess.run(msame_cmd, capture_output=True, shell=False)
            log = out.stdout.decode('utf-8')
            match = re.search("Inference average time without first time: (.{1,350}) ms", log)
            if not match or out.returncode:
                raise RuntimeError("msame inference failed!\n{}".format(log))
            try:
                time = float(match.group(1))
            except ValueError as err:
                logger.error("Received an unexpected runtime value from msame's log.")
                raise RuntimeError("Please check your msame inference runtime environment.") from err
            
        elif self.tool == 'pyacl':
            from pyacl.acl_infer import AclNet, init_acl

            device_id = cfg.get('device_id', None) or 0
            try:
                init_acl(device_id)
            except Exception as err:
                logger.error("acl init failed! error message: {}".format(err))
                raise RuntimeError("acl init failed! {}".format(err)) from err
            try:
                net = AclNet(model_path=cfg['model'], device_id=device_id)
            except Exception as err:
                logger.error("load model failed! error message: {}".format(err))
                raise RuntimeError("load model failed! {}".format(err)) from err
            time = 0.0
            for _ in range(loop):
                data = in_queue.get()
                if len(data) < 2:  # include file_name and data
                    logger.error("data len less than 2! data should include label and data!")
                    raise RuntimeError("input params error len={}".format(len(data)))
                try:
                    outputs, exe_time = net(data[1])
                except Exception as err:
                    logger.error("acl infer failed! error message: {}".format(err))
                    raise RuntimeError("acl infer error len={}".format(len(data))) from err
                out_queue.put([data[0], outputs])
                time += exe_time
        logger.debug("inference end")
        return time
