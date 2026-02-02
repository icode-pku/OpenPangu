# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
from random import Random
from datetime import datetime, timedelta, timezone


class RandomNameSequence(object):
    characters = "abcdefghijklmnopqrstuvwxyz0123456789_"

    def __init__(self):
        self._rng = None
        self._rng_pid = None

    def __iter__(self):
        return self
    
    def __next__(self):
        return ''.join(self.rng.choices(self.characters, k=8))

    @property
    def rng(self):
        cur_pid = os.getpid()
        if cur_pid != getattr(self, '_rng_pid', None):
            self._rng = Random()
            self._rng_pid = cur_pid
        return self._rng


def get_timestamp():
    cst_timezone = timezone(timedelta(hours=8))
    current_time = datetime.now(cst_timezone)
    return current_time.strftime("%Y%m%d%H%M%S")
