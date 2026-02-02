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
import operator

from .version import Version


def and_(a, b):
    "Same as a and b."
    return a and b


def or_(a, b):
    "Same as a or b."
    return a or b


class Evaluator:
    OPS = {
        # 一元运算符
        'not': (7, operator.not_),  # 最高优先级
        
        # 算术运算符
        '**': (6, operator.pow),    # 右结合
        '*': (5, operator.mul), '/': (5, operator.truediv),
        '//': (5, operator.floordiv), '%': (5, operator.mod),
        '+': (4, operator.add), '-': (4, operator.sub),
        
        # 比较运算符 (所有同级)
        '==': (3, operator.eq), '!=': (3, operator.ne),
        '>': (3, operator.gt), '<': (3, operator.lt),
        '>=': (3, operator.ge), '<=': (3, operator.le),
        
        # 逻辑/位运算符
        '&': (2, operator.and_), 'and': (2, and_),
        '|': (1, operator.or_), 'or': (1, or_)
    }

    _TOKEN_REGEX = re.compile(
        r'(?P<FUNC>\b(?:int|str|float|Version)\b)'
        r'|(?P<NUM>-?\b\d+(\.\d*)?\b)'
        r'|(?P<OP>==|!=|>=|<=|\*\*|//|>|<|and|or|not|[+\-*/%])'
        r'|(?P<PARENTHESE>[)(])'
        r'|(?P<COMMA>,)'
        r'|(?P<STR>\'[^\']+\')'
        r'|(?P<NONE>\bNone\b)'
        r'|(?P<BOOL>\b(?:(?:t|T)rue|(?:f|F)alse)\b)'
        r'|(?P<SKIP>\s+)'
    )
    
    _FUNC_MAP = {
        'float': float, 'int': int,
        'str': str, 'Version': Version
    }

    @classmethod
    def evaluate(cls, expr: str):
        if isinstance(expr, list):
            return [cls.evaluate(ex) for ex in expr]
        elif isinstance(expr, dict):
            return {k: cls.evaluate(ex) for k, ex in expr.items()}
        elif isinstance(expr, str):
            return cls._evaluate(expr)
        else:
            return expr

    @classmethod
    def _evaluate(cls, expr: str):
        rpn = cls._convert_rpn(expr)
        try:
            result = cls._evaluate_rpn(rpn)
        except (ValueError, TypeError, ZeroDivisionError):
            return expr
        return result

    @classmethod
    def _convert_rpn(cls, expr: str):
        stack = []
        output = []
        nargs = 1

        for mo in cls._TOKEN_REGEX.finditer(expr):
            kind = mo.lastgroup
            val = mo.group(kind)

            if kind == 'SKIP':
                continue

            if kind == "FUNC":
                stack.append(val)
            elif kind == "OP":
                cls._process_op(val, stack, output)
            elif kind == "PARENTHESE":
                cls._process_parenthese(val, stack, output, nargs)
            elif kind == "COMMA":
                while stack and stack[-1] != '(':
                    num_func_args += 1
                    output.append(stack.pop())
            elif kind == "NUM":
                output.append(float(val) if '.' in val else int(val))
            elif kind == "STR":
                output.append(val.strip("'"))
            elif kind == "NONE":
                output.append(None)
            elif kind == "BOOL":
                output.append(val == "true" or val == "True")
            else:
                output.append(val)
    
        while stack:
            if stack[-1] in ('(', ')'):
                raise ValueError("Mismatched parentheses in expression")
            output.append(stack.pop())

        return output

    @classmethod
    def _process_parenthese(cls, val, stack, output, nargs):
        """ 
        If val is '(', push to stack; otherwise pop all ops in stack between the previous ')'
        """
        if val == '(':
            stack.append(val)
            return

        while stack and stack[-1] != '(':
            output.append(stack.pop())

        if not stack:
            raise ValueError("Mismatched parentheses in expression")

        stack.pop()
        if stack and stack[-1] in cls._FUNC_MAP:
            func = stack.pop()
            fn = cls._FUNC_MAP[func]
            output.append((fn, nargs))
            nargs = 1

    @classmethod
    def _process_op(cls, op, stack, output):
        """
        Each op has its precedence and function, pop all the op in the stack where its prec
        is higher than the current op (that means they calculate first), and push current op
        to stack
        """
        prec = cls.OPS[op][0]
        while stack and stack[-1] in cls.OPS and cls.OPS[stack[-1]][0] >= prec:
            output.append(stack.pop()) # pop last op to output and push itself to stack
        stack.append(op)

    @classmethod
    def _evaluate_rpn(cls, rpn):
        stack = []
        for token in rpn:
            if isinstance(token, tuple):
                if len(token) != 2:
                    stack.append(token)
                    continue

                fn, nargs = token
                if not callable(fn):
                    stack.append(token)
                    continue

                if len(stack) < nargs:
                    raise ValueError("Not enough args for function")
                
                args = [stack.pop() for _ in range(nargs)][::-1]
                stack.append(fn(*args))

            elif isinstance(token, Version):
                stack.append(token)
            elif token == "not": # unary
                a = stack.pop()
                stack.append(cls.OPS[token][1](a))
            elif token in cls.OPS:
                b = stack.pop()
                a = stack.pop()
                stack.append(cls.OPS[token][1](a, b))
            else:
                stack.append(token)

        if len(stack) != 1:
            raise ValueError(f"Invalid RPN evaluation: {rpn}")

        return stack[0]
