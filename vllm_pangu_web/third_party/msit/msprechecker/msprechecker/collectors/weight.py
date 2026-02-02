# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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
import os
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

from msguard import Rule, where, Path
from msguard.security import walk_s

from .base import BaseCollector


class WeightCollector(BaseCollector):
    def __init__(self, error_handler=None, *, weight_dir=None, chunk_size=None):
        super().__init__(error_handler)
        self.error_handler.type = "weight"
        self.weight_dir = weight_dir
        self.chunk_size = chunk_size

    @staticmethod
    def _calculate_hash256(filepath, chunk_size):
        sha256_hash = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                sha256_hash.update(data)
        return sha256_hash.hexdigest()

    def _validate_inputs(self):
        if self.weight_dir is None:
            self.error_handler.add_error(
                filename=__file__,
                function='_validate_inputs',
                lineno=45,
                what="未传入权重目录",
                reason=f"未传入权重目录前不应该调用 'WeightCollector'"
            )
            return False
        valid_sizes = [size * 1024 ** 2 for size in [32, 64, 128, 256]]
        if self.chunk_size not in valid_sizes:
            self.error_handler.add_error(
                filename=__file__,
                function='_validate_inputs',
                lineno=55,
                what="'chunk_size' 不符合要求",
                reason=f"'chunk_size' 需要为 {valid_sizes}"
            )
            return False
        return True

    def _get_tensor_files(self, tensor_suffix):
        max_weight_size = 10 * 1024 ** 3
        weight_rule = where(
            os.getuid() == 0,
            Path.is_file(),
            Path.is_file() & ~Path.has_soft_link() &
            Path.is_readable() & ~Path.is_writable_to_group_or_others() &
            Path.is_consistent_to_current_user() & Path.is_size_reasonable(size_limit=max_weight_size),
            description="current user is root"
        )

        tensor_files = [
            path
            for path in walk_s(self.weight_dir, file_rule=weight_rule)
            if os.path.isfile(path) and path.endswith(tensor_suffix)
        ]
        if not tensor_files:
            self.error_handler.add_error(
                filename=__file__,
                function='_get_tensor_files',
                lineno=67,
                what="权重目录下没有找到符合条件的权重路径",
                reason=f"工具只会收集 {tensor_suffix!r} 结尾的权重路径，且符合安全要求"
            )
        return tensor_files

    def _process_futures(self, futures, tensor_id_pattern):
        results = {}
        for future in as_completed(futures):
            tensor_file = futures[future]
            tensor_basename = os.path.basename(tensor_file)
            m = tensor_id_pattern.search(tensor_basename)
            tensor_id = m.group(1) if m else tensor_basename
            try:
                result = future.result()
            except Exception as e:
                self.error_handler.add_error(
                    filename=__file__,
                    function='_process_futures',
                    lineno=82,
                    what=f"计算文件 sha256 哈希失败: {tensor_file!r}",
                    reason=str(e)
                )
                result = "Unknown"
            results[tensor_id] = result
        return results

    def _collect_data(self):
        results = {}
        if not self._validate_inputs():
            return results

        tensor_suffix = '.safetensors'
        tensor_files = self._get_tensor_files(tensor_suffix)
        if not tensor_files:
            return results

        max_workers = min(len(tensor_files), os.cpu_count() or 1)
        tensor_id_pattern = re.compile(
            r'(\d{5})-of-\d{5}' + re.escape(tensor_suffix)
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._calculate_hash256, tensor_file, self.chunk_size): tensor_file
                for tensor_file in tensor_files
            }
            results = self._process_futures(futures, tensor_id_pattern)

        return results
