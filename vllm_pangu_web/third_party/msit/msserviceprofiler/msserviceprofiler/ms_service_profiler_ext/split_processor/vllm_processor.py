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

from .base_processor import BaseFrameworkProcessor


class VllmProcessor(BaseFrameworkProcessor):
    batch_start_name = "batchFrameworkProcessing"
    batch_end_name = "forward"
    http_start_name = "httpReq"
    http_end_name = "httpRes"
    key_name = "forward"
    all_time_name = "AllTime"
    http_list = ["httpReq", "encode", "decodeEnd", "httpRes"]
    name_list = [batch_start_name, "preprocess", batch_end_name]
    filter_list = [http_start_name, http_end_name, all_time_name]
    name_list = name_list + http_list

    @classmethod
    def initialize(cls, args):
        cls.args = args