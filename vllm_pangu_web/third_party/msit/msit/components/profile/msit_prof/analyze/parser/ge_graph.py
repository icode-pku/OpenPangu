# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd. All rights reserved.
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
import json

from components.utils.file_open_check import ms_open
from components.utils.constants import JSON_FILE_MAX_SIZE


def get_all_subgraph(graph_path):
    with ms_open(graph_path, max_size=JSON_FILE_MAX_SIZE) as file:
        ge_graph = json.load(file)
        for graph in ge_graph.get('graph', []):
            yield graph