# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
import string
from msit_llm.common.log import logger

CODE_CHAR = string.printable.replace("\r", "")  # For getting rid of Chinese char and windows `\r`


def print_spelling(param, info="", level="debug"):
    param = param.get_children() if hasattr(param, "get_children") else param
    message = info + "[" + ", ".join([ii.spelling for ii in param]) + "]"
    if level.lower() == "debug":
        logger.debug(message)
    else:
        logger.info(message)


def print_update_info(insert_contents, insert_start, insert_end, cur_id=None):
    message = f"insert_start: {insert_start}, insert_end: {insert_end}, insert_contents: {insert_contents}"
    if cur_id is not None:
        message += f", cur_id: {cur_id}"
    logger.debug("Current update: " + message)


def check_libclang_so():
    import clang
    from clang import cindex

    libclang_so_path = os.path.join(os.path.dirname(clang.__file__), "native", "libclang.so")
    if os.path.exists(libclang_so_path):
        cindex.Config.set_library_file(libclang_so_path)
    else:
        logger.warning("libclang so: %r not found, may meet error lately." % libclang_so_path)


def get_args_and_options():
    import platform    
    from clang import cindex

    atb_home_path = os.getenv("ATB_HOME_PATH", "")
    ascend_toolkit_home = os.getenv("ASCEND_TOOLKIT_HOME", "")
    atb_speed_compile_path = os.path.dirname(os.path.dirname(os.getenv("ATB_SPEED_HOME_PATH", "")))

    cur_platform = platform.machine() + "-" + platform.system()  # like "aarch64-linux"
    include_pathes = [
        os.path.join(atb_home_path, "include"),
        os.path.join(ascend_toolkit_home, cur_platform, "include"),
        atb_speed_compile_path,
        os.path.join(atb_speed_compile_path, "core", "include"),
        os.path.join(atb_speed_compile_path, "3rdparty", "nlohmannJson", "include"),
    ]
    args = ["-fsyntax-only"]
    args.extend(["-I " + include_path for include_path in include_pathes])
    options = cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    return args, options


def filter_chinese_char(contents):
    return "".join(filter(lambda ii: ii in CODE_CHAR, contents))


def update_contents(contents, updates):
    updates = sorted(updates, key=lambda xx: xx[0], reverse=True)
    for insert_start, insert_end, insert_contents in updates:
        contents = contents[:insert_start] + insert_contents + contents[insert_end:]
    return contents
