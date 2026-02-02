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

from collections import deque


class Traverser:
    @staticmethod
    def traverse(data):
        if not isinstance(data, (dict, list)):
            raise TypeError(f"Expected 'data' to be dict or list. Got {type(data).__name__} instead.")
            
        queue = deque()
        queue.append((data, ""))

        visited_nodes = {}

        while queue:
            node, path = queue.popleft()

            if path:
                visited_nodes[path] = node

            if isinstance(node, dict):
                Traverser._traverse_dict(node, path, queue)
            elif isinstance(node, list):
                Traverser._traverse_list(node, path, queue)

        return visited_nodes

    @staticmethod
    def _traverse_dict(node: dict, path: str, queue: deque):
        for k, v in node.items():
            new_path = k if not path else path + "." + k
            queue.append((v, new_path))

    @staticmethod
    def _traverse_list(node: list, path: str, queue: deque):
        for i, v in enumerate(node):
            new_path = f"[{i}]" if not path else path + f"[{i}]"
            queue.append((v, new_path))
