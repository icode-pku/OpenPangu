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

from collections import defaultdict
from ..utils import get_handler, ErrorHandler, ErrorType


class Comparator:
    MISSING_VALUE = "missing"

    def __init__(self, *, error_handler: ErrorHandler = None):
        self.error_handler = error_handler or get_handler(ErrorType.ERR_COMPARE)

    @staticmethod
    def _flatten(data, visited_path, file_path, path=()):
        stack = [(data, path)]
        
        while stack:
            current_data, current_path = stack.pop()
            
            if isinstance(current_data, dict):
                for k, v in current_data.items():
                    new_path = current_path + (k,)
                    if k == "npuDeviceIds":
                        visited_path[new_path][file_path] = v
                    else:
                        stack.append((v, new_path))
            elif isinstance(current_data, list):
                for i, v in enumerate(reversed(current_data)):
                    new_path = current_path + (i,)
                    stack.append((v, new_path))
            else:
                visited_path[current_path][file_path] = current_data

    @staticmethod
    def _unflatten(flat_dict):
        root = {}
        
        for path, value in flat_dict.items():
            node = root
            for i, part in enumerate(path[:-1]):
                next_part = path[i + 1]
                if isinstance(next_part, int):
                    if part not in node:
                        node[part] = [{}]
                    while len(node[part]) <= next_part:
                        node[part].append({})
                else:
                    if part not in node:
                        node[part] = {}
                node = node[part]
            node[path[-1]] = value

        return root

    @staticmethod
    def _all_values_equal(values, all_paths) -> bool:
        ref_value = next(iter(values.values()))
        
        return all(
            path in values and values[path] == ref_value
            for path in all_paths
        )

    def compare(self, path_to_data):
        flat_diff = {}
            
        for path, data in path_to_data.items():
            for conf_type, conf_data in data.items():
                flat_diff.setdefault(conf_type, defaultdict(dict))
                conf_diff = flat_diff[conf_type]

                self._flatten(conf_data, conf_diff, path, ())

        filtered_flat_diff = {
            conf_type: self._unflatten({
                path: {
                    file_path: values.get(file_path, self.MISSING_VALUE)
                    for file_path in path_to_data
                }
                for path, values in conf_diff.items()
                if not self._all_values_equal(values, path_to_data)
            })
            for conf_type, conf_diff in flat_diff.items()
        }

        for collect_type, values in filtered_flat_diff.items():
            self.error_handler.add_error(collect_type, values)
                    
        return self.error_handler
