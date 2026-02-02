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
config.read('../../config/config.ini')

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

debug_sub_tasks = [{
    "name": "compare",
    "help_info": "one-click network-wide accuracy analysis of golden models.",
    "module": "msquickcmp.__main__",
    "attr": "get_compare_cmd_ins"
}, {
    "name": "dump",
    "help_info": "one-click dump model ops inputs and outputs.",
    "module": "msquickcmp.__main__",
    "attr": "get_dump_cmd_ins"
}]

debug_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in debug_sub_tasks
]

setup(
    name='msit-compare',
    version='8.2.0',
    description='This tool enables one-click network-wide accuracy analysis of gold model.',
    long_description="",
    long_description_content_type='text/markdown',
    url=config.get('URL', 'msit_debug_compare_url'),
    packages=find_packages(),
    package_data={'': ['LICENSE', 'install.sh', 'libsaveom.so', '*.cpp']},
    license='Apache-2.0',
    keywords='compare',
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
        'Topic :: Software Development'
    ],
    python_requires='>=3.7',
    entry_points={
        'debug_sub_task': debug_sub_task_entry_points,
        'msit_sub_task_installer': ['msit-compare=msquickcmp.__install__:CompareInstall'],
    },
)
