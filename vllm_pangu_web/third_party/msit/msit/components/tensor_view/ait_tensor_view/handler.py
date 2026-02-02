# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import splitext

import torch

from ait_tensor_view.atb import read_atb_data, write_atb_data
from ait_tensor_view.print_stat import print_stat
from components.utils.log import logger
from components.utils.util import safe_torch_load


def replace(in_path: str, out_path: str) -> str:
    in_ext = splitext(in_path)[1]
    out_ext = splitext(out_path)[1]

    if in_ext and not out_ext:
        out_path += in_ext

    return out_path


def handle_tensor_view(args):
    in_ext = splitext(args.bin)[1]
    if in_ext == ".bin":
        tensor = read_atb_data(args.bin)
    else:
        tensor = safe_torch_load(args.bin, map_location="cpu")

    logger.info(f"source tensor shape: {tensor.shape}")

    if args.operations:
        logger.info("Operations start")

        for op in args.operations:
            logger.info(f"{op.name} starts, current tensor shape: {tensor.shape}")
            tensor = op.process(tensor)
            logger.info(f"{op.name} ends, current tensor shape: {tensor.shape}")

        logger.info("Operations end")

    print_stat(tensor)

    if args.print:
        logger.info("\n%s", tensor)

    if args.output:
        out_path = args.output
        out_ext = splitext(out_path)[1]

        try:
            if not out_ext or (out_ext != ".bin" and out_ext != ".pth"):
                out_path += in_ext
            if out_path.endswith(".bin"):
                write_atb_data(tensor, out_path)
            else:
                torch.save(tensor, out_path)

            logger.info(f'Tensor saved successfully to {out_path}')
        except Exception as e:
            logger.error(f"Error saving Tensor: {e}")
