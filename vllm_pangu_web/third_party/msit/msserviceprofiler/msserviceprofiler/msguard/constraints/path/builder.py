# -*- coding: utf-8 -*-
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

import re

from .path import PathConstraint


class PathMeta(type):
    lower_after_upper_pattern = re.compile(
        r'(?<=[a-z])' # match the position where right after a lower letter
        r'(?=[A-Z])' # and before a upper letter
    )

    def __new__(cls, name, bases, namespace):
        new_class = super().__new__(cls, name, bases, namespace)

        for sub_cls in PathConstraint.__subclasses__():
            sub_cls_name = sub_cls.__name__

            # replace the zero width to underscore _, that makes AbCd to Ab_Cd
            snake_name = cls.lower_after_upper_pattern.sub('_', sub_cls_name).lower()
            setattr(new_class, snake_name, sub_cls)

        return new_class

    def __getattr__(cls, name):
        for sub_cls in PathConstraint.__subclasses__():
            sub_cls_name = sub_cls.__name__
            snake_name = cls.lower_after_upper_pattern.sub('_', sub_cls_name).lower()
            if snake_name == name:
                setattr(cls, snake_name, sub_cls)
                return sub_cls

        return super().__getattr__(name)


class Path(metaclass=PathMeta):
    pass
