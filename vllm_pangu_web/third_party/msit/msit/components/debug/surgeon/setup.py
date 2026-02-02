# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

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



debug_sub_tasks = [
    {
        "name": "surgeon",
        "help_info": "surgeon tool for onnx modifying functions.",
        "module": "auto_optimizer.ait_main",
        "attr": "get_cmd_instance"
    }
]

debug_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in debug_sub_tasks
]

setup(
    name='msit-surgeon',
    version='8.2.0',
    description='auto optimizer',
    long_description="",
    long_description_content_type='text/markdown',
    url=config.get('URL', 'msit_url'),
    packages=find_packages(),
    package_data={'': ['LICENSE', 'model.cfg']},    

    license='Apache-2.0',
    keywords='auto optimizer',
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
    extras_require={
        'inference': [
            (
                f"aclruntime @ git+{config.get('URL', 'msit_url')}"
                '#egg=aclruntime&subdirectory=msit/components/benchmark/backend'
            ),
            (
                f"ais_bench @ git+{config.get('URL', 'msit_url')}"
                '#egg=ais_bench&subdirectory=msit/components/benchmark/'
            ),
            'pillow >= 9.0.0',
            'tqdm >= 4.63.0',
        ],
        'simplify': ['onnx-simplifier >= 0.3.6'],
    },
    python_requires='>=3.7',
    entry_points={
        'console_scripts': ['auto_optimizer=auto_optimizer.__main__:cli'],
        'debug_sub_task': debug_sub_task_entry_points,
        'msit_sub_task_installer': ['msit-surgeon=auto_optimizer.__install__:SurgeonInstall'],
    },
)
