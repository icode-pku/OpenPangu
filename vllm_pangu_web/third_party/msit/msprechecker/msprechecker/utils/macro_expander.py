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
from functools import partial


class ExpandError(Exception):
    def __init__(self, expr):
        self.expr = expr
        super().__init__(f"Cannot expand {expr}")


class MacroExpander:
    DOT = "."
    VAR_REGEX = re.compile(r"\$\{([\w_.\[\]/]+?)\}")

    @classmethod
    def expand(cls, expr, context, visited_path):
        if isinstance(expr, dict):
            return {k: cls.expand(v, context, visited_path) for k, v in expr.items()}
        elif isinstance(expr, list):
            return [cls.expand(item, context, visited_path) for item in expr]
        elif isinstance(expr, str):
            full_path = cls.VAR_REGEX.sub(partial(cls._expand_var, context=context, visited_path=visited_path), expr)
            return full_path
        else:
            return expr

    @classmethod
    def _expand_var(cls, match_object: re.Match, context, visited_path):
        path = match_object.group(1)
        if not path:
            raise ExpandError(path)

        full_path = cls._build_full_path(path, context)
        if full_path not in visited_path:
            raise ExpandError(full_path)

        val = visited_path[full_path]
        if isinstance(val, str):
            return repr(val)

        return str(val)

    @classmethod
    def _build_full_path(cls, path: str, context: str) -> str:
        if path == cls.DOT:
            return context

        if path.startswith(cls.DOT):
            full_path = context + path
            comps = full_path.split('.')
        
            stack = []
            for comp in comps:
                if comp:
                    stack.append(comp)
                elif not stack:
                    raise ExpandError(full_path)
                else:
                    stack.pop()

            return '.'.join(stack)

        return path
