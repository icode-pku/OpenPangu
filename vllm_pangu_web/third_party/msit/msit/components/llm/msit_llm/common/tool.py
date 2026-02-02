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
import random
import array
import argparse
import torch
import numpy as np

from msit_llm.common.constant import ATTR_END, ATTR_OBJECT_LENGTH, LCCL_DETERMINISTIC, HCCL_DETERMINISTIC, \
    ATB_MATMUL_SHUFFLE_K_ENABLE, ATB_LLM_LCOC_ENABLE, PYTHON_HASH_SEED
from msit_llm.common.log import logger
from msit_llm.common.utils import check_input_path_legality, check_data_file_size, load_file_to_read_common_check
from components.utils.file_open_check import ms_open
from components.utils.constants import TENSOR_MAX_SIZE
from components.utils.util import safe_int


class TensorBinFile:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.dtype = 0
        self.format = 0
        self.dims = []
        self.dtype_dict = {
            0: torch.float32, 
            1: torch.float16, 
            2: torch.int8, 
            3: torch.int32, 
            9: torch.int64,
            12: torch.bool, 
            27: torch.bfloat16
        }

        self._parse_bin_file()

    def get_data(self):
        if self.dtype not in self.dtype_dict:
            logger.error("Unsupported dtype %s", self.dtype)
            raise ValueError("Unsupported dtype {}".format(self.dtype))
        dtype = self.dtype_dict.get(self.dtype)
        tensor = torch.frombuffer(array.array('b', self.obj_buffer), dtype=dtype)
        tensor = tensor.view(self.dims)
        return tensor

    def _parse_bin_file(self):
        self.file_path = load_file_to_read_common_check(self.file_path)
        with ms_open(self.file_path, "rb", max_size=TENSOR_MAX_SIZE) as fd:
            file_data = fd.read()

        begin_offset = 0
        for i, byte in enumerate(file_data):
            if byte != ord("\n"):
                continue
            line = file_data[begin_offset: i].decode("utf-8")
            begin_offset = i + 1
            fields = line.split("=")
            if len(fields) != 2:
                raise ValueError("Unsupported tensorbin file, we need data format is 'attr_name=attr_value'.")
            attr_name = fields[0]
            attr_value = fields[1]
            if attr_name == ATTR_END:
                self.obj_buffer = file_data[i + 1:]
                break
            elif attr_name.startswith("$"):
                self._parse_system_attr(attr_name, attr_value)
            else:
                self._parse_user_attr(attr_name, attr_value)

    def _parse_system_attr(self, attr_name, attr_value):
        if attr_name == ATTR_OBJECT_LENGTH:
            self.obj_len = safe_int(attr_value)

    def _parse_user_attr(self, attr_name, attr_value):
        if attr_name == "dtype":
            self.dtype = safe_int(attr_value)
        elif attr_name == "format":
            self.format = safe_int(attr_value)
        elif attr_name == "dims":
            self.dims = attr_value.split(",")
            self.dims = [safe_int(dim) for dim in self.dims]


def read_atb_data(file_path):
    file_path = check_input_path_legality(file_path)

    if file_path.endswith(".bin"):
        if check_data_file_size(file_path):
            bin_tensor = TensorBinFile(file_path)
            data = bin_tensor.get_data()
            return data

    raise ValueError("Tensor file path must be end with .bin.")



def seed_all(seed=2024):
    if not isinstance(seed, int):
        raise argparse.ArgumentTypeError("%s is not an int." % seed)
    
    os.environ[LCCL_DETERMINISTIC] = "1"
    os.environ[HCCL_DETERMINISTIC] = "true"
    os.environ[ATB_MATMUL_SHUFFLE_K_ENABLE] = "0"
    os.environ[ATB_LLM_LCOC_ENABLE] = "0"

    os.environ[PYTHON_HASH_SEED] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode=True)

    try:
        import torch_npu
    except ImportError:
        is_npu = False 
    else:
        is_npu = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.benchmark = False
    if is_npu and torch_npu.npu.is_available():
        torch_npu.npu.manual_seed(seed)
        torch_npu.npu.manual_seed_all(seed)

    logger.info(f"Enable deterministic computation sucess! current seed is {seed}.")
