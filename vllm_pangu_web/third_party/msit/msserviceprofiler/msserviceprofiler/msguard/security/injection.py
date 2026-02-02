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

import os
import re
import pickle
import inspect
from io import BytesIO

from .io import open_s
from .exception import CSVInjectionError, PickleInjectionError


CSV_INJECTION_PATTERN = re.compile(r'^[＋－＝％＠\+\-=%@]|;[＋－＝％＠\+\-=%@]')


# csv injection
def is_safe_csv_value(value: str) -> bool:
    if not isinstance(value, str):
        return True

    try:
        float(value)
    except ValueError:
        return not bool(CSV_INJECTION_PATTERN.search(value))

    return True


def sanitize_csv_value(value: str, *, replace=False) -> str:
    if is_safe_csv_value(value):
        return value
    
    if replace:
        return "'" + value

    err_msg = f'Malicious value is not allowed to be written into the CSV: {value}'
    raise CSVInjectionError(err_msg)


# pickle injection
class SafeUnpickler(pickle.Unpickler):
    def __init__(self, file, call_back_fn=None, *, fix_imports=True,
                 encoding="ASCII", errors="strict"):
        if call_back_fn is not None:
            self._validate_callback(call_back_fn)
        super().__init__(file, fix_imports=fix_imports,
                         encoding=encoding, errors=errors)
        self.call_back_fn = call_back_fn if call_back_fn else self.default_safe_callback

    @staticmethod
    def default_safe_callback(module: str, name: str) -> bool:    
        safe_combinations = {
            'builtins': {'int', 'float', 'str', 'list', 'tuple', 'dict', 'set', 'frozenset', 'bool'},
            'collections': {'OrderedDict', 'defaultdict', 'deque'},
            'datetime': {'date', 'datetime', 'time', 'timedelta', 'timezone'},
            'numpy': {'ndarray', 'dtype', 'float64', 'int64'},
            'pandas': {'DataFrame', 'Series', 'Index'},
            'math': {'inf', 'nan'},
        }
        
        base_module = module.split('.')[0]
        allowed_names = safe_combinations.get(base_module, set())
        return name in allowed_names
    
    @staticmethod
    def _validate_callback(call_back_fn) -> None:
        if not callable(call_back_fn):
            raise TypeError(
                f"Expected 'call_back_fn' to be callable. Got {type(call_back_fn).__name__} instead."
            )
        
        sig = inspect.signature(call_back_fn)
        if len(sig.parameters) != 2:
            raise ValueError("Callback must accept exactly 2 parameters (module_name, global_name)")

    def find_class(self, module_name, global_name):
        try:
            if not self.call_back_fn(module_name, global_name):
                raise PickleInjectionError(f"Attempting to load a malicious object: {module_name}.{global_name}.")

            return super().find_class(module_name, global_name)
        except PickleInjectionError:
            raise
        except Exception as e:
            raise RuntimeError(f"Security verification failed for {module_name}.{global_name}") from e


def pickle_load_s(file, *, fn=None):
    if isinstance(file, (str, bytes, os.PathLike)):
        with open_s(file, 'rb') as f:
            return SafeUnpickler(f, call_back_fn=fn).load()

    return SafeUnpickler(file, call_back_fn=fn).load()


def pickle_loads_s(data, *, fn=None):
    return pickle_load_s(BytesIO(data), fn=fn)
