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
from configparser import ConfigParser
from setuptools import setup, find_packages

config = ConfigParser()
config.read('../../config/config.ini')

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

msit_sub_tasks = [
    {
    "name": "elb",
    "help_info": "Load balancing expert optimization strategy.",
    "module": "elb.__main__",
    "attr": "get_cmd_instance"
}
]

msit_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in msit_sub_tasks
]

setup(
    name='msit-elb',
    version='8.2.0',
    description='Deepseek model load balancing affinity expert optimization in static and dynamic scenarios',
    packages=find_packages(), 
    license='Apache-2.0',
    keywords='elb',
    install_requires=required,
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Developers',
        'License :: Apache-2.0 Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    python_requires='>=3.9',
    entry_points={
        'msit_sub_task': msit_sub_task_entry_points,
        'msit_sub_task_installer': ['msit-elb=elb.__install__:ExpertLoadBalanceInstall'],
    },
)