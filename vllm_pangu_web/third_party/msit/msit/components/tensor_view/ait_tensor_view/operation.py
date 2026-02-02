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

import re

import torch

SLICE_PATTERN = re.compile(r"\.\.\.|-?\d+|(-?\d+)?:(-?\d+)?(:?-?\d+)?")


def convert_slice(s: str) -> slice:
    parts = [n.strip() for n in s.split(":")]
    size = len(parts)

    if size == 2:
        start, end = parts
        start = int(start) if start else None
        end = int(end) if end else None
        return slice(start, end, None)
    elif size == 3:
        start, end, step = parts
        start = int(start) if start else None
        end = int(end) if end else None
        step = int(step) if step else None
        return slice(start, end, step)
    else:
        raise SyntaxError(f"'{s}' is not valid slice string")


class SliceOperation:
    def __init__(self, slice_str: str):
        content = slice_str[1:-1].split(",")

        parts = []

        for c in content:
            c = c.strip()
            if c != "":
                if not SLICE_PATTERN.match(c):
                    raise SyntaxError(f"slice string is invalid: '{slice_str}', part: '{c}'")
                parts.append(c)

        if len(parts) == 0:
            raise SyntaxError("slice string can not be empty")

        self.slice_raw = slice_str
        self.parts = parts
        self.name = f"SliceOperation: {parts}"

    def process(self, prev: torch.Tensor) -> torch.Tensor:
        size = len(self.parts)
        shape = prev.shape
        shape_size = len(shape)

        if size > shape_size:
            raise ValueError(f"number of dimensions[{self.slice_raw}] is bigger than the shape size[{shape_size}]")
        slices = []

        for i, part in enumerate(self.parts):
            if ":" in part:
                slices.append(convert_slice(part))
            elif part == "...":
                slices.append(Ellipsis)
            else:
                index = int(part)
                dim_size = shape[i]
                if index >= dim_size or -1 * index > dim_size:
                    raise IndexError(f"Index out of range, dim[{i}]_size is {dim_size} while index={index}")
                slices.append(index)

        if len(slices) == 1 and isinstance(slices[0], int):
            slices = slices[0]

        return prev[slices]


class PermuteOperation:
    def __init__(self, permute_str: str):
        parts = [int(n.strip()) for n in permute_str[1:-1].split(",") if n != ""]
        self.permute_raw = permute_str
        self.parts = parts
        self.name = f"PermuteOperation: {parts}"

    def process(self, prev: torch.Tensor) -> torch.Tensor:
        shape = prev.shape
        n = len(shape)

        if len(self.parts) != len(shape):
            raise ValueError(f"'{self.permute_raw}' has no enough dimension")

        self.check_permute_elements()
        self.check_permute_range(n)

        return prev.permute(self.parts)

    def check_permute_elements(self):
        if not all(isinstance(x, int) for x in self.parts):
            raise ValueError(f"All elements in {self.permute_raw} should be integer, but got {self.parts}")
        if len(set(self.parts)) != len(self.parts):
            raise ValueError(f"{self.permute_raw} contains duplicate dimensions!")

    def check_permute_range(self, n: int):
        if not all(0 <= x < n for x in self.parts):
            raise ValueError(f"{self.permute_raw}: not all dimensions are between 0 and {n - 1}")
