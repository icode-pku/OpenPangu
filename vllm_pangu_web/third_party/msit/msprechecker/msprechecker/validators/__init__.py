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

__all__ = ["get_validator"]


_validator_registry = {}


def register_validator(validator_type: str, validator_cls: type):
    _validator_registry[validator_type] = validator_cls


def _lazy_register_validators():
    from .range import RangeValidator
    from .cmp import (
        GreaterThanValidator, LessThanValidator, EqualValidator,
        GEValidator, LEValidator, NEValidator
    )
    from .enum import EnumValidator
    from .path import PathValidator
    from .docker import DockerValidator

    aliases = {
        "range": RangeValidator,
        "enum": EnumValidator, "in": EnumValidator,
        "eq": EqualValidator, "==": EqualValidator,
        "lt": LessThanValidator, "<": LessThanValidator,
        "gt": GreaterThanValidator, ">": GreaterThanValidator,
        "ge": GEValidator, ">=": GEValidator,
        "le": LEValidator, "<=": LEValidator,
        "ne": NEValidator, "!=": NEValidator,
        "path": PathValidator,
        "docker": DockerValidator
    }

    for key, cls in aliases.items():
        register_validator(key, cls)


def get_validator(validator_type: str) -> type:
    if not _validator_registry:
        _lazy_register_validators()

    if validator_type not in _validator_registry:
        raise ValueError(f"Unknown validator type: {validator_type}")
        
    return _validator_registry[validator_type]
