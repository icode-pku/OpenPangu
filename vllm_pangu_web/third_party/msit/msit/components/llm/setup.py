# Copyright (c) 2023-2025 Huawei Technologies Co., Ltd.
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

import site
import os

from configparser import ConfigParser
from setuptools import setup, find_packages

config = ConfigParser()
config.read('../config/config.ini')

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

opchecker_lib_src = []
for root, _, files in os.walk('components/llm/msit_llm/opcheck/test_framework/'):
    opchecker_lib_src.append((os.path.join("/", root), [os.path.join(root, f) for f in files]))

msit_sub_tasks = [
    {
        "name": "llm",
        "help_info": "Large Language Model(llm) Debugger Tools.",
        "module": "msit_llm.__main__",
        "attr": "get_cmd_instance"
    }
]

msit_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in msit_sub_tasks
]

if not site.getsitepackages():
    raise RuntimeError('There are no global site-packages directories.')

setup(
    name='msit_llm',
    version='8.2.0',
    description='Debug tools for large language model(llm)',
    url=config.get('URL', 'msit_llm_url'),
    packages=find_packages(),
    package_data={'': ['*.sh', '*.cpp', '*.h', '*.txt']},
    license='Apache-2.0',
    keywords='msit_llm',
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
    data_dir=f"{site.getsitepackages()[0]}",
    data_files=opchecker_lib_src,
    include_package_data=True,
    python_requires='>=3.7',
    entry_points={
        'msit_sub_task': msit_sub_task_entry_points,
        'msit_sub_task_installer': ['msit-llm=msit_llm.__install__:LlmInstall'],
    },
)
