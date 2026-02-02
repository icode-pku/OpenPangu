# Copyright (c) 2023-2025 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup, find_packages  # type: ignore

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

msit_sub_tasks = [
    {
    "name": "profile",
    "help_info": "get profiling data of a given programma",
    "module": "msit_prof.main_cli",
    "attr": "get_cmd_instance"
}
]

msit_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in msit_sub_tasks
]

setup(
    name='msit-profile',
    version='8.2.0',
    description='msprof tool',
    url='msit_msprof url',
    packages=find_packages(),
    keywords='msit_msprof tool',
    install_requires=required,
    python_requires='>=3.7',
    entry_points={
        'msit_sub_task': msit_sub_task_entry_points,
        'msit_sub_task_installer': ['msit-profile=msit_prof.__install__:MsProfInstall'],
    }
)
