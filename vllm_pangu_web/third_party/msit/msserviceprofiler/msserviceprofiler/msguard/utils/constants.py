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


CONFIG_FILE_MAX_SIZE = 1 * 1024 * 1024 # work for .ini config file
TEXT_FILE_MAX_SIZE = 5 * 1024 * 1024 * 1024 # work for txt, csv, py
JSON_FILE_MAX_SIZE = 1024 * 1024 * 1024
DB_MAX_SIZE = 50 * 1024 * 1024 * 1024
LOG_FILE_MAX_SIZE = 5 * 1024 * 1024


EXT_SIZE_MAPPING = {
    '.db': DB_MAX_SIZE,
    ".ini": CONFIG_FILE_MAX_SIZE,
    '.csv': TEXT_FILE_MAX_SIZE,
    '.txt': TEXT_FILE_MAX_SIZE,
    '.json': JSON_FILE_MAX_SIZE,
    '.log': LOG_FILE_MAX_SIZE
}

DEFAULT_MAX_FILES = 10000
DEFAULT_MAX_DEPTHS = 100
DEFAULT_FILE_MODE = 0o640
DEFAULT_DIR_MODE = 0o750
VALID_OPEN_MODES = {'r', 'w', 'x', 'a', 'b', 't', '+'}
