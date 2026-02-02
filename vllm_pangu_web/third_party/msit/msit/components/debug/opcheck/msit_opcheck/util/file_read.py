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
import os
import subprocess

from components.utils.util import filter_cmd
from components.utils.security_check import check_input_path_legality


def get_msaccucmp_path():
    # 加载CANN TOOLKIT环境变量，该变量在分包安装下不存在，需要手动进行配置
    cann_path = os.environ.get('ASCEND_TOOLKIT_HOME', "/usr/local/Ascend/ascend-toolkit/latest")
    msaccucmp_path = os.path.join(cann_path, "tools", "operator_cmp", "compare", "msaccucmp.py")
    msaccucmp_path = check_input_path_legality(msaccucmp_path)
    return msaccucmp_path


def execute_convert_npy_command(command):
    try:
        command = filter_cmd(command)
        result = subprocess.run(command, shell=False, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error while converting bin data to npy: {result.stderr.strip()}"
    except Exception as e:
        return f"Error while converting bin data to npy: {e}"
    

def convert_ge_dump_file_to_npy(input_path, npy_path):
    # input_path可以指定单个文件或者是目录
    msaccucmp_path = get_msaccucmp_path()
    command = ["python3", msaccucmp_path, "convert", "-d", input_path, "-out", npy_path]
    return_msg = execute_convert_npy_command(command)
    if "Error while converting bin data to npy:" in return_msg:
        raise RuntimeError("Execute msaccucmp convert failed, Please check CANN Environment.")
    

def get_ascbackend_ascgraph(ascgen_path):
    ascgraph_list = []
    for element in os.listdir(ascgen_path):
        # 当前不支持对fusedgraph进行解析构图
        if element.endswith(".py") and not element.startswith("autofuse_fused"):
            ascgraph_list.append(element)
    return ascgraph_list