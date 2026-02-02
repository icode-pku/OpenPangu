# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from enum import Enum


class DataType(Enum):
    MINDIE_SUMED_CSV = 0
    MINDIE_SPLITED_CSV = 1
    MINDIE_SPLITED_CSV_WITH_TOPK = 2
    VLLM_SUMED_TENSOR = 3
    UNKNOWN_TYPE = 999


class BaseDataLoader:
    def __init__(self, input_args):
        self.input_args = input_args
        self.data_type = DataType.UNKNOWN_TYPE
