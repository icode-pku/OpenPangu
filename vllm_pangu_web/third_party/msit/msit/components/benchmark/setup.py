# -*- coding: utf-8 -*-
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
from configparser import ConfigParser
from setuptools import setup, find_packages  # type: ignore

config = ConfigParser()
cur_dir = os.path.dirname(os.path.abspath(__file__))  # benchmark
components_dir = os.path.dirname(cur_dir)  # components
config_ini_path = os.path.join(components_dir, 'config', 'config.ini')
config.read(config_ini_path)

msit_sub_tasks = [
    {
        "name": "benchmark",
        "help_info": "benchmark tool to get performance data including latency and throughput",
        "module": "msit_benchmark.__main__",
        "attr": "get_cmd_instance",
    }
]

msit_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}" 
    for t in msit_sub_tasks
]

setup(
    name="msit-benchmark",
    version='8.2.0',
    description="msit benchmark tool",
    url=config.get('URL', 'msit_url'),
    packages=find_packages(),
    package_data={"": ["LICENSE", "README.md", "requirements.txt", "install.bat", "install.sh", "*.cpp", "*.h"]},
    keywords="msit benchmark tool",
    classifiers=[
        "Development Status :: Alpha",
        "Intended Audience :: Developers",
        "License :: Apache-2.0 Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    python_requires=">=3.7",
    entry_points={
        "msit_sub_task": msit_sub_task_entry_points,
        "msit_sub_task_installer": ["msit-benchmark=msit_benchmark.__install__:BenchmarkInstall"],
    },
)
