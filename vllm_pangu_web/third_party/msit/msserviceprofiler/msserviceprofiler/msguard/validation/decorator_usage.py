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

import inspect
import functools

from ..constraints import InvalidParameterError, make_constraint


def validate_parameter_constraint(parameter_constraint, params, caller_name):
    for param_name, param_val in params.items():
        if param_name not in parameter_constraint:
            continue

        constraint = parameter_constraint[param_name]
        constraint = make_constraint(constraint)

        if not constraint.is_satisfied_by(param_val):
            raise InvalidParameterError(param_name, caller_name, constraint, param_val)


def validate_params(parameter_constraint):

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_sig = inspect.signature(func)

            params = func_sig.bind(*args, **kwargs)
            params.apply_defaults()

            ignore_params = [
                p.name
                for p in func_sig.parameters.values()
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ] + ["self", "cls"]
            params = {k: v for k, v in params.arguments.items() if k not in ignore_params}

            validate_parameter_constraint(
                parameter_constraint, params, func.__qualname__
            )

            return func(*args, **kwargs)
        return wrapper

    return decorator
