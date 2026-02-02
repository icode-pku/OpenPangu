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
try:
    from ms_service_profiler.utils.log import logger, LOG_LEVELS, set_logger, set_log_level
except ModuleNotFoundError:
    import logging

    LOG_LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
        "critical": logging.CRITICAL,
    }


    def set_log_level(level="info"):
        if level.lower() in LOG_LEVELS:
            logger.setLevel(LOG_LEVELS.get(level.lower()))
        else:
            logger.warning("Set %s log level failed.", level)


    def set_logger(msit_logger):
        msit_logger.propagate = False
        msit_logger.setLevel(logging.INFO)
        if not msit_logger.handlers:
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s")
            stream_handler.setFormatter(formatter)
            msit_logger.addHandler(stream_handler)


    logger = logging.getLogger("msServiceProfiler")
    set_logger(logger)


def confirmation_interaction(prompt):
    confirm_pattern = re.compile(r'y(?:es)?', re.IGNORECASE)
    
    try:
        user_action = input(prompt)
    except Exception:
        return False
    
    return bool(confirm_pattern.match(user_action))
