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

import json
import shutil
import logging
from abc import ABC, abstractmethod

from msguard.security import open_s

from ..utils import ErrorHandler, CollectErrorHandler, ConfigErrorHandler, CheckErrorHandler, CompareErrorHandler


class ErrorDisplayStrategy(ABC):
    COLOR_RESET = "\033[0m"
    COLOR_GRAY = "\033[38;5;247m"
    COLOR_RED = "\033[91m"
    COLOR_GREEN = "\033[92m"
    COLOR_YELLOW = "\033[93m"
    COLOR_BLUE = "\033[94m"
    COLOR_CYAN = "\033[96m"
    COLOR_MAGENTA = "\033[95m"

    def __init__(self):
        self.logger = self._init_logger()

    @staticmethod
    def _init_logger():
        local_logger = logging.getLogger(__name__)
        local_logger.setLevel(logging.INFO)
        local_logger.propagate = False

        if not local_logger.handlers:
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            stream_handler.setFormatter(formatter)
            local_logger.addHandler(stream_handler)
        return local_logger

    @abstractmethod
    def display(self, error_handler: ErrorHandler):
        pass

    def _print_title(self, title: str, fillchar="="):
        cols, _ = shutil.get_terminal_size()
        self.logger.info(f" {title} ".center(cols, fillchar))

    def _print_success(self):
        self.logger.error('\033[92mAll checks passed\033[0m')


class CollectErrorDisplay(ErrorDisplayStrategy):
    def display(self, error_handler):
        title = f"{error_handler.type} collect report".upper()
        self._print_title(title)

        if not error_handler.errors:
            self._print_success()
            return

        for error in error_handler:
            context = error.context
            filename = context.filename
            lineno = context.lineno
            function = context.function
            what = context.what

            self._print_title(f"{filename}({lineno}){function}()")
            self.logger.info(
                f"%s {self.COLOR_YELLOW}%s{self.COLOR_RESET}. %s",
                error.severity,
                what,
                error.reason
            )


class CheckErrorDisplay(ErrorDisplayStrategy):
    def display(self, error_handler):
        title = f"{error_handler.type} check report".upper()
        self._print_title(title)

        if not error_handler.errors:
            self._print_success()
            return

        for error in error_handler:
            context = error.context
            path = context.path
            actual = context.actual
            expected = context.expected

            self.logger.error(f"\033[96m-- {path}\033[0m {error.severity}")
            self.logger.error(
                f"    - \033[1;91m{actual}\033[0m "
                f"\033[1;97m->\033[0m "
                f"\033[1;92m{expected}\033[0m "
                f"\033[38;5;247m<-- {error.reason}\033[0m"
            )


class ConfigErrorDisplay(ErrorDisplayStrategy):
    def display(self, error_handler):
        title = f"{error_handler.type} check report".upper()
        self._print_title(title)

        if not error_handler.errors:
            self._print_success()
            return

        lineno_mapping = dict()
        max_lineno = 0

        for error in error_handler:
            context = error.context
            context_lines = context.context_lines
            lineno = context.lineno

            lineno_mapping.update(context_lines)
            lineno_mapping[lineno] = error

            if isinstance(lineno, int):
                max_lineno = max(max_lineno, len(str(lineno)))

        for lineno in sorted(lineno_mapping):
            value = lineno_mapping[lineno]
            if isinstance(value, str): 
                self.logger.error(f"{self.COLOR_GRAY}{lineno:>{max_lineno}}: %s{self.COLOR_RESET}", value)
                continue

            context = value.context
            path = context.path
            expected = context.expected
            actual = context.actual
            start_col = context.start_col
            severity = value.severity
            reason = value.reason

            if isinstance(lineno, float):
                lineno = "?"

            line_prefix = " " * start_col
            actual_line = f'{self.COLOR_RESET}{json.dumps(path)}: {json.dumps(actual)}{self.COLOR_RESET}'
            self.logger.error(
                f"{self.COLOR_RED}{lineno:>{max_lineno}}:{self.COLOR_RESET} {line_prefix}{actual_line} {severity}"
            )

            caret_indent = max_lineno + 2 + start_col + 1
            caret_line = " " * caret_indent + f"{self.COLOR_RED}{'^' * len(path)}{self.COLOR_RESET}"
            self.logger.error(caret_line)

            suggestion_indent = max_lineno + 2 + start_col
            suggestion_line = (
                " " * suggestion_indent +
                f'{self.COLOR_MAGENTA}{json.dumps(path)}: {json.dumps(expected)}{self.COLOR_RESET} <--- {reason}'
            )
            self.logger.error("%s", suggestion_line)


class EnvCheckErrorDisplayDecorator(ErrorDisplayStrategy):
    """Decorator for special handling of env errors."""
    def __init__(self, decorated_strategy: ErrorDisplayStrategy):
        super().__init__()
        self.decorated = decorated_strategy

    @staticmethod
    def _generate_env_script(error_handler):
        activate_cmds = []
        deactivate_cmds = []
        for error in error_handler:
            context = error.context
            var = context.path
            expected = context.expected
            actual = context.actual
            reason = error.reason

            if expected is None:
                activate_cmds.append(f"unset {var} # {reason}")
            else:
                activate_cmds.append(f'export {var}="{expected}" # {reason}')

            if actual == "<missing>":
                deactivate_cmds.append(f"unset {var}")
            else:
                deactivate_cmds.append(f'export {var}="{actual}"')

        script = (
            "#!/bin/bash\n"
            "# --------------------------------------------------\n"
            "# 环境变量管理脚本 (直接 source 执行)\n"
            "# 使用方式:\n"
            "#   source msprechecker_env.sh    # 应用预期配置\n"
            "#   source msprechecker_env.sh 0  # 还原为原始状态\n"
            "# --------------------------------------------------\n\n"
            'if [ "$1" = "0" ]; then\n'
            "    {deactivate}\n"
            "else\n"
            "    {activate}\n"
            "fi\n"
        ).format(
            deactivate='\n    '.join(deactivate_cmds),
            activate='\n    '.join(activate_cmds)
        )
        return script

    def display(self, error_handler):
        self.decorated.display(error_handler)
        if not error_handler.errors:
            return

        script_content = self._generate_env_script(error_handler)

        with open_s('./msprechecker_env.sh', 'w') as f:
            f.write(script_content)

        self.logger.info(
            "\n# To apply the environment changes, use\n#\n"
            "#    $ source ./msprechecker_env.sh\n#\n"
            "# To restore the previous environment changes, use\n#\n"
            "#    $ source ./msprechecker_env.sh 0"
        )


class CompareErrorDisplay(ErrorDisplayStrategy):
    def display(self, error_handler):
        for diff in error_handler:
            title = f"{diff.key} diff report".upper()
            self._print_title(title)

            if not diff.values:
                self.logger.error('\033[92mThere is no difference found.\033[0m')
                continue

            self.logger.error(json.dumps(diff.values, indent=2))


class ErrorDisplayStrategyFactory:
    _strategy_map = {
        CollectErrorHandler: CollectErrorDisplay,
        CheckErrorHandler: CheckErrorDisplay,
        ConfigErrorHandler: ConfigErrorDisplay,
        CompareErrorHandler: CompareErrorDisplay
    }

    @classmethod
    def get_strategy(cls, error_handler: ErrorHandler) -> ErrorDisplayStrategy:
        for handler_type, strategy in cls._strategy_map.items():
            if isinstance(error_handler, handler_type):
                base_strategy = strategy()
                # 如果是env类型，用装饰器包装基础策略
                if isinstance(error_handler, CheckErrorHandler) and error_handler.type == "env":
                    return EnvCheckErrorDisplayDecorator(base_strategy)
                return base_strategy
        raise TypeError

    @classmethod
    def register_strategy(cls, error_type, strategy: ErrorDisplayStrategy):
        cls._strategy_map[error_type] = strategy
