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

import torch
from tabulate import tabulate
from components.utils.log import logger


VALID_DTYPES = [
    torch.float,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.complex32,
    torch.complex64,
    torch.complex128
]


def print_stat(tensor: torch.Tensor):
    tmp = tensor
    if tensor.dtype not in VALID_DTYPES:
        tmp = tensor.clone().to(torch.float32)

    table = [
        ["min", "max", "mean", "std", "var"],
        [tmp.min(), tmp.max(), tmp.mean(), tmp.std(), tmp.var()]
    ]

    logger.info("\n%s", tabulate(table, tablefmt="grid"))
    