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

import logging


class ANSIColoredFormatter(logging.Formatter):
    COLORS = {
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'RESET': '\033[0m'        # 重置
    }

    def format(self, record):
        message = super().format(record)
        if record.levelname in self.COLORS:
            message = f"{self.COLORS[record.levelname]}{message}{self.COLORS['RESET']}"
        return message


def get_logger():
    logger = logging.getLogger("msprechecker")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = ANSIColoredFormatter("%(message)s")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


global_logger = get_logger()
