# -*- coding: utf-8 -*-
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

import os
import stat


class Const:

    ONLY_READ = stat.S_IRUSR | stat.S_IRGRP  # 0O440

    # bin path
    FAST_QUERY_BIN = os.path.join('tools', 'ms_fast_query', 'ms_fast_query.py')

    # error detail
    ERR_UNSUPPORT = 'Op is unsupported.'
    ERR_OPP_NOT_EXIST = 'Opp not exist.'
