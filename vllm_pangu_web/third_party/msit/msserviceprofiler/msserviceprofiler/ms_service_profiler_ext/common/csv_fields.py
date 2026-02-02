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


class BaseCSVFields(object):
    METRIC = "Metric"
    AVG = "Average"
    MAX = "Max"
    MIN = "Min"
    P50 = "P50"
    P90 = "P90"
    P99 = "P99"

    columns = (METRIC, AVG, MAX, MIN, P50, P90, P99)


class BatchCSVFields(BaseCSVFields):
    PREFILL_BATCH_NUM = "prefill_batch_num"
    DECODE_BATCH_NUM = "decode_batch_num"
    PREFILL_EXEC_TIME = "prefill_exec_time (ms)"
    DECODE_EXEC_TIME = "decode_exec_time (ms)"

    PATH_NAME = "batch_summary.csv"

    metrics = (PREFILL_BATCH_NUM, DECODE_BATCH_NUM, PREFILL_EXEC_TIME, DECODE_EXEC_TIME)


class RequestCSVFields(BaseCSVFields):
    FIRST_TOKEN_LATENCY = "first_token_latency (ms)"
    SUBSEQUENT_TOKEN_LATENCY = "subsequent_token_latency (ms)"
    TOTAL_TIME = "total_time (ms)"
    EXEC_TIME = "exec_time (ms)"
    WAITING_TIME = "waiting_time (ms)"
    INPUT_TOKEN_NUM = "input_token_num"
    GENERATED_TOKEN_NUM = "generated_token_num"

    PATH_NAME = "request_summary.csv"

    metrics = (
        FIRST_TOKEN_LATENCY, SUBSEQUENT_TOKEN_LATENCY, TOTAL_TIME,
        EXEC_TIME, WAITING_TIME, INPUT_TOKEN_NUM, GENERATED_TOKEN_NUM
    )


class ServiceCSVFields(BaseCSVFields):
    VALUE = "Value"
    TOTAL_INPUT_TOKEN_NUM = "total_input_token_num"
    TOTAL_GENERATED_TOKEN_NUM = "total_generated_token_num"
    GENERATE_TOKEN_SPEED = "generate_token_speed (token/s)"
    GENERATE_ALL_TOKEN_SPEED = "generate_all_token_speed (token/s)"

    PATH_NAME = "service_summary.csv"

    metrics = (
        TOTAL_INPUT_TOKEN_NUM, TOTAL_GENERATED_TOKEN_NUM,
        GENERATE_TOKEN_SPEED, GENERATE_ALL_TOKEN_SPEED
    )
    columns = (BaseCSVFields.METRIC, VALUE)
