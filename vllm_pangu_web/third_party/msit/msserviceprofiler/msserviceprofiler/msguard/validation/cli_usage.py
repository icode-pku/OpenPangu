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

import argparse
from functools import wraps

from ..constraints import BaseConstraint, InvalidParameterError
from .decorator_usage import validate_params


def validate_args(constraint: BaseConstraint, *, fall_back_fn=None):
    """
    Usage: parser.add_argument("xx", type=validate_args(xxx))

    pass the `wrapper` to argparse, pass `arg` to `arg_check`and catch the error msg
    """
    def arg_check(arg: str):
        if not constraint.is_satisfied_by(arg):
            invalid_param_error = InvalidParameterError(
                'arg', arg_check.__qualname__,
                constraint, arg
            )
            error_msg = invalid_param_error.build_error_message()

            if not fall_back_fn:
                raise argparse.ArgumentTypeError(error_msg)
            
            fall_back_fn(arg)
        return arg

    @wraps(arg_check)
    def wrapper(arg):
        return arg_check(arg)

    return wrapper
