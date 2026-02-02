# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

config = ConfigParser()
config.read("./components/config/config.ini")

abs_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(abs_path, "requirements.txt")) as f:
    required = f.read().splitlines()

msit_sub_tasks = [
    {
        "name": "debug",
        "help_info": "debug a wide variety of model issues",
        "module": "components.debug.__init__",
        "attr": "debug_task",
    }
]

msit_sub_task_entry_points = []
for t in msit_sub_tasks:
    name = t.get('name')
    help_info = t.get('help_info')
    module = t.get('module')
    attr = t.get('attr')

    entory_point = f"{name}:{help_info} = {module}:{attr}"
    msit_sub_task_entry_points.append(entory_point)


class DevelopWithShUmask(develop):
    def run(self):
        cur_umask, sh_mode = 0o027, 0o750
        os.umask(cur_umask)
        develop.run(self)
        for filepath in self.get_outputs():
            if filepath.endswith(".sh"):
                os.chmod(filepath, sh_mode)  # Has to be manually 750, cannot be stat with umask


class InstallWithShUmask(install):
    def run(self):
        cur_umask, sh_mode = 0o027, 0o750
        os.umask(cur_umask)
        install.run(self)
        for filepath in self.get_outputs():
            if filepath.endswith(".sh"):
                os.chmod(filepath, sh_mode)  # Has to be manually 750, cannot be stat with umask


setup(
    name="msit",
    version="8.2.0",
    description="msIT, MindStudio Inference Tools",
    long_description_content_type="text/markdown",
    url=config.get("URL", "msit_url"),
    packages=find_packages(),
    package_data={
        "": [
            "LICENSE",
            "README.md",
            "*.txt",
            "*.bat",
            "*.sh",
            "*.cpp",
            "*.h",
            "*.ini",
        ]
    },
    data_files=[("", ["requirements.txt"])],
    license="Apache-2.0",
    keywords="msit",
    python_requires=">=3.7",
    install_requires=required,
    entry_points={
        "console_scripts": ["msit=components.__main__:main"],
        "msit_sub_task": msit_sub_task_entry_points,
    },
    cmdclass={"develop": DevelopWithShUmask, "install": InstallWithShUmask},
)
