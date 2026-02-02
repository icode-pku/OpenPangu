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


import functools


def validate_parameters_by_func(parameter_constraints, in_class=False):
    if not isinstance(parameter_constraints, dict):
        raise TypeError(f"Parameter constraints expects dict, but got {type(parameter_constraints).__name__} instead.")

    if not parameter_constraints:
        raise ValueError("Parameter constraints should not be empty.")

    if not all(isinstance(key, str) for key in parameter_constraints.keys()):
        raise ValueError("Key of the parameter constraints only supports string.")

    if not all(isinstance(val, (tuple, list)) for val in parameter_constraints.values()):
        raise ValueError("Value of the parameter constraints only supports tuple or list.")

    def decorator(func):

        def _check_constraint(arg, constraint_name, constraint):
            if not constraint:
                return

            for check_item in constraint:

                if callable(check_item):
                    check_name = check_item.__name__
                    
                    test_result = None
                    try:
                        test_result = check_item(arg)
                    except Exception as e:
                        msg = f"In the running function `{func.__name__}`, the argument `{constraint_name}`, \
                            whose value is `{arg}`, is invalid as it has not been passed through the designated \
                            constraints."
                        raise RuntimeError(msg) from e

                    try:
                        test_result = True if test_result is None else bool(test_result)
                    except Exception as e:
                        msg = f"The result from the designated constraints `{check_name}` \
                            can not be interpreted as bool."
                        raise RuntimeError(msg) from e

                    if not test_result:
                        msg = f"In the running function `{func.__name__}`, the argument `{constraint_name}`, \
                            whose value is `{arg}`, is invalid as it has not been passed through the designated \
                            constraints." 
                        raise RuntimeError(msg)

                else:
                    msg = f"Provided `{check_item}` that associated with key `{constraint_name}` is invalid: \
                        Not callable."
                    raise TypeError(msg)
            
        if in_class:
            @functools.wraps(func)
            def wrapper(self_or_cls, *args, **kwargs):
                constraints_iterator = iter(parameter_constraints.items())

                if args:
                    for arg, (constraint_name, constraint) in zip(args, constraints_iterator):
                        _check_constraint(arg, constraint_name, constraint)

                if kwargs:
                    for (_, arg_val), (constraint_name, constraint) in zip(kwargs.items(), constraints_iterator):
                        _check_constraint(arg_val, constraint_name, constraint)

                return func(self_or_cls, *args, **kwargs)

            return wrapper
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            constraints_iterator = iter(parameter_constraints.items())

            if args:
                for arg, (constraint_name, constraint) in zip(args, constraints_iterator):
                    _check_constraint(arg, constraint_name, constraint)

            if kwargs:
                for (_, arg_val), (constraint_name, constraint) in zip(kwargs.items(), constraints_iterator):
                    _check_constraint(arg_val, constraint_name, constraint)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# if no type matches, the function should also check type again, which is not desired
def validate_parameters_by_type(parameter_constraints, in_class=False):
    """Easy type checking

    Automatically check the arguments of a function based on given constraints `parameter_constraints` in the runtime.
    `parameter_constraints` is a user-defined dictionary and users should be responsible for the match of key and the
    real argument name.

    Self-defined functions should only take one argument. By default, if the function is not passed, it will raise an
    exception. If the test is complicated, or related to other variable, or user wants to other exception messages, 
    user should take care it on their own.

    in_class: take care of self and cls, static method should not use this option.

    Users also should take care of `*arg` and `**kwargs` on their own. This is only designed for explicit argument
    cases.
    """

    if not isinstance(parameter_constraints, dict):
        raise TypeError(f"Parameter constraints expects dict, but got {type(parameter_constraints).__name__} instead.")

    if not parameter_constraints:
        raise ValueError("Parameter constraints should not be empty.")

    if not all(isinstance(key, str) for key in parameter_constraints.keys()):
        raise ValueError("Key of the parameter constraints only supports string.")

    if not all(isinstance(val, (tuple, list)) for val in parameter_constraints.values()):
        raise ValueError("Value of the parameter constraints only supports tuple or list.")

    def decorator(func):

        def _check_constraint(arg, constraint_name, constraint):
            if not constraint:
                return

            type_list = []

            for check_item in constraint:
                if isinstance(check_item, type):
                    check_name = check_item.__name__
                    type_list.append(check_name)

                    # if that is the one
                    if isinstance(arg, check_item):
                        return

                elif isinstance(check_item, type(None)):
                    type_list.append(None)

                    if arg is None:
                        return
                        
                else:
                    msg = f"Provided `{check_item}` that associated with key `{constraint_name}` is invalid. \
                        Only types are allowed."
                    raise TypeError(msg)

            msg = f"In the running function `{func.__name__}`, the argument `{constraint_name}`, \
                whose value is `{arg}`, is invalid, where `{type_list}` is expected, \
                but got `{type(arg).__name__}` instead."
            raise TypeError(msg)

        if in_class:
            @functools.wraps(func)
            def wrapper(self_or_cls, *args, **kwargs):
                constraints_iterator = iter(parameter_constraints.items())

                if args:
                    for arg, (constraint_name, constraint) in zip(args, constraints_iterator):
                        _check_constraint(arg, constraint_name, constraint)

                if kwargs:
                    for (_, arg_val), (constraint_name, constraint) in zip(kwargs.items(), constraints_iterator):
                        _check_constraint(arg_val, constraint_name, constraint)

                return func(self_or_cls, *args, **kwargs)

            return wrapper
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            constraints_iterator = iter(parameter_constraints.items())

            if args:
                for arg, (constraint_name, constraint) in zip(args, constraints_iterator):
                    _check_constraint(arg, constraint_name, constraint)

            if kwargs:
                for (_, arg_val), (constraint_name, constraint) in zip(kwargs.items(), constraints_iterator):
                    _check_constraint(arg_val, constraint_name, constraint)

            return func(*args, **kwargs)

        return wrapper

    return decorator
