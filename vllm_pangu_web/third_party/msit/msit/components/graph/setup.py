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

from configparser import ConfigParser
from setuptools import setup, find_packages  # type: ignore

config = ConfigParser()
config.read('../config/config.ini')

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

msit_sub_tasks = [
    {
        "name": "graph",
        "help_info": "Graph analyze tool to analyze GE graph",
        "module": "msit_graph.__main__",
        "attr": "get_cmd_instance",
    }
]

msit_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}" 
    for t in msit_sub_tasks
]

setup(
    name='msit-graph',
    version='8.2.0',
    description='graph analyze tool',
    url=config.get('URL', 'msit_url'),
    packages=find_packages(),
    package_data={'': ['*.sh', '*.cpp', '*.h', '*.txt']},
    license='Apache-2.0',
    keywords='graph analyze tool',
    install_requires=required,
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Developers',
        'License :: Apache-2.0 Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    python_requires='>=3.7',
    entry_points={
        'msit_sub_task': msit_sub_task_entry_points,
        'msit_sub_task_installer': ['msit-graph=msit_graph.__install__:GraphAnalyzeInstall'],
    },
)
