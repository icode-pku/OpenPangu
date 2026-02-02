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

import sys
import time
import logging


class ProcessBarStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + "\r")  # 用 \r 代替默认的 \n
            self.flush()
        except Exception:
            self.handleError(record)


class SimpleProgressBar:
    def __init__(self, iterable, desc=None, total=None):
        self.iterable = iterable
        self.desc = desc or ""
        self.total = total if total is not None else len(iterable)
        self.current = 0
        self.start_time = time.time()
        self.logger = self._init_logger()

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)
        self.logger.info("\n")

    @staticmethod
    def _init_logger():
        local_logger = logging.getLogger(__name__)
        local_logger.setLevel(logging.INFO)
        local_logger.propagate = False

        if not local_logger.handlers:
            handler = ProcessBarStreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            local_logger.addHandler(handler)

        return local_logger

    def update(self, n=1):
        self.current += n
        self._print_progress()

    def _print_progress(self):
        progress = self.current / self.total
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        percent = progress * 100

        # 计算剩余时间
        elapsed_time = time.time() - self.start_time
        if progress > 0:
            remaining_time = (elapsed_time / progress) * (1 - progress)
        else:
            remaining_time = 0

        trailing_space = " " * 4 # invisible but better for progress bar
        self.logger.info(
            f"\r{self.desc} |{bar}| {percent:.1f}% [{self.current}/{self.total}] "
            f"ETA: {remaining_time:.1f}s{trailing_space}"
        )
