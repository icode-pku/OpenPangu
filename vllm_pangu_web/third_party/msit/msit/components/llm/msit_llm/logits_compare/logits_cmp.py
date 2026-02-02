# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd. All rights reserved.
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
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

import csv
import torch

from msit_llm.common.log import logger
from msit_llm.common.utils import load_file_to_read_common_check
from components.utils.util import safe_torch_load
from components.utils.security_check import ms_makedirs
from components.utils.constants import CSV_FILE_MAX_SIZE
from components.utils.cmp_algorithm import cosine_similarity, kl_divergence, l1_norm
from components.utils.file_open_check import ms_open, sanitize_csv_value


CMP_ROWS_LEN = 10
FILE_NAME = "file_name"
KEY = "key"
KEY_INDEX = 1
TOKEN_ID = "token_id"
TOKEN_ID_INDEX = 2
COS_SIMILARITY = "cosine_similarity"
COS_SIMILARITY_INDEX = 3
KL_DIVERGENCE = "kl_divergence"
KL_DIVERGENCE_INDEX = 4
L1_NORM = "l1_norm"
L1_NORM_INDEX = 5
ULP_MAX_DIFF = "ulp_max_diff"
ULP_MAX_DIFF_INDEX = 6
ULP = "ulp"
ULP_INDEX = 7
PASSED = "passed"
PASSED_INDEX = 8
CMP_FAIL_REASON = "cmp_fail_reason"
CMP_FAIL_REASON_INDEX = 9
ULP_DTYPE_MIN_VALUE = {
    "bf16": 2 ** -7,
    "fp16": 2 ** -10,
    "fp32": 2 ** -23
}
CMP_ALG_MAP = {
    "cosine_similarity": cosine_similarity,
    "kl_divergence": kl_divergence,
    "l1_norm": l1_norm
}


@dataclass
class RowData:
    file_name: str = "NA"
    key: str = "NA"
    token_id: str = "NA"
    cosine_sim: str = "NA"
    kl_div: str = "NA"
    l1_res: str = "NA"
    ulp_diff: str = "NA"
    ulp: str = "NA"
    passed: str = "NA"
    cmp_fail_reason: str = "NA"


def compute_ulp(golden_data: torch.Tensor, my_data: torch.Tensor, dtype: str):
    '''
    Function: compute ulp
    '''
    golden_data = golden_data.float()
    my_data = my_data.float()
    golden_max_value = torch.max(golden_data)
    my_max_value = torch.max(my_data)
    logger.debug(f"Max golden and my tensor value: {golden_max_value}, {my_max_value}")
    logits_diff = torch.abs(golden_max_value - my_max_value)
    if my_max_value == 0:
        return None, None, "max value is 0, cannot compute ulp"
    e = torch.floor(torch.log2(my_max_value))
    ulp = (2 ** e) * (ULP_DTYPE_MIN_VALUE[dtype])
    return logits_diff.item(), ulp.item(), ""


def get_timestamp():
    cst_timezone = timezone(timedelta(hours=8))
    current_time = datetime.now(cst_timezone)
    return current_time.strftime("%Y%m%d%H%M%S")


def parse_key_id(filename):
    '''
    解析文件名中的key和token id, humaneval-X数据集独特处理
    '''
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')
    
    # humaneval-X支持的编程语言
    langs = {'cpp', 'go', 'java', 'js', 'python'}
    found_langs = [part for part in parts if part in langs]
    
    if found_langs:
        # 新格式处理：取最后出现的语言标识
        lang = found_langs[-1]
        
        # 逆向查找两个数字部分
        nums = []
        for part in reversed(parts):
            if part.isdigit():
                nums.append(part)
                if len(nums) == 2:
                    break
        if len(nums) < 2:
            return None, None, f"missing_digits_after_{lang}"
        
        try:
            key = f"{lang}/{nums[1]}"  # 倒数第二个数字
            token_id = int(nums[0])          # 最后一个数字
            return key, token_id, None
        except ValueError:
            return None, None, f"invalid_digits_{nums}"
    else:
        # 其他数据集
        if len(parts) < 2:
            return None, None, "invalid logits file name"
        
        key_part, id_part = parts[-2], parts[-1]
        try:
            return int(key_part), int(id_part), None
        except ValueError:
            return None, None, f"invalid_format_{key_part}_{id_part}"


def load_logits_from_file(logits_path):
    if not isinstance(logits_path, str) or not logits_path.endswith('.pth'):
        logger.error("Invalid logits path, only path with suffix '.pth' is allowed: '%r'" % logits_path)
        raise ValueError
    logits = safe_torch_load(logits_path)
    return logits.squeeze(0)


def compare_tensor(golden_data, my_data, dtype):
    row_data, fail_messages = {}, []

    logger.debug(f"Check the golden_data shape: {golden_data.shape}, type: {golden_data.dtype}")
    logger.debug(f"Check the my_data shape: {my_data.shape}, type: {my_data.dtype}")
    # 检查tensor的shape是否一致、是否存在NAN或inf
    tensor_pass, message = check_tensor(golden_data, my_data)
    if not tensor_pass:
        logger.debug(f"check_tensor failed: {message}")
        row_data[CMP_FAIL_REASON] = message
        return row_data

    for name, cmp_func in list(CMP_ALG_MAP.items()):
        result, message = cmp_func(golden_data, my_data)
        row_data[name] = result
        if len(message) > 0:
            fail_messages.append(message)
        logger.debug(f"Calculate {name}, result is {result}")
    ulp_max_diff, ulp, compute_ulp_fail_reason = compute_ulp(golden_data, my_data, dtype)
    row_data[ULP_MAX_DIFF] = ulp_max_diff
    row_data[ULP] = ulp
    fail_messages.append(compute_ulp_fail_reason)
    row_data[CMP_FAIL_REASON] = " ".join(fail_messages)
    logger.debug(f"Compare tensor result: {row_data[COS_SIMILARITY]}, {row_data[KL_DIVERGENCE]},"
                 f"{row_data[L1_NORM]}, {row_data[ULP_MAX_DIFF]}, {row_data[ULP]}")
    return row_data


def check_tensor(golden_data, my_data):
    tensor_pass = True
    fail_reasons = []

    # 检验golden tensor和my tensor的shape是否一致
    if golden_data.shape != my_data.shape:
        fail_reasons.append("data shape doesn't match.")
        tensor_pass = False
    # 检验golden_data中是否存在NAN或者inf
    if not torch.all(torch.isfinite(golden_data)):
        fail_reasons.append("golden_data includes NAN or inf.")
        tensor_pass = False
    # 检验my_data中是否存在NAN或者inf
    if not torch.all(torch.isfinite(my_data)):
        fail_reasons.append("my_data includes NAN or inf.")
        tensor_pass = False
    return tensor_pass, " ".join(fail_reasons)


# 混合类型排序逻辑
def sort_key(row):
    if len(row) < 3:
        raise RuntimeError("Sort row error, row must have cols: 'key' and 'token_id'")
    key_val = row[KEY_INDEX]
    fid_val = row[TOKEN_ID_INDEX] if isinstance(row[TOKEN_ID_INDEX], int) else float('inf')
    
    first_priority, second_priority, third_priority, fourth_priority = 0, 1, 2, 3
    # 类型判断
    if isinstance(key_val, int):  # 旧格式
        return (first_priority, '', key_val, fid_val)
    elif isinstance(key_val, str) and '/' in key_val:  # 新格式
        try:
            lang, num = key_val.split('/', 1)
            return (second_priority, lang, int(num), fid_val)
        except ValueError as e:
            return (fourth_priority, '', float('inf'), float('inf'))
    else:  # 无效格式
        return (third_priority, '', float('inf'), float('inf'))


def create_rows(row_data: RowData):
    return [row_data.file_name, row_data.key, row_data.token_id, row_data.cosine_sim, row_data.kl_div,
            row_data.l1_res, row_data.ulp_diff, row_data.ulp, row_data.passed, row_data.cmp_fail_reason]


class LogitsComparison:
    def __init__(self, args):
        self.golden_path = args.golden_path
        self.my_path = args.my_path
        self.cosine_similarity = args.cosine_similarity
        self.kl_divergence = args.kl_divergence
        self.l1_norm = args.l1_norm
        self.output_dir = args.output_dir
        self.dtype = args.dtype

    def compare_logits(self):
        logger.info("Compare the logits of 'golden' and 'my', and calculate the result")
        if self.my_path is None or self.golden_path is None:
            raise ValueError("Please ensure that both --my-path and --golden-path "
                             "are provided.")
        if not (os.path.isdir(self.my_path) and os.path.isdir(self.golden_path)):
            raise RuntimeError("Please ensure that both --my-path and --golden-path "
                               "are dir")
        golden_files = {f for f in os.listdir(self.golden_path) if f.endswith('.pth')}
        my_files = {f for f in os.listdir(self.my_path) if f.endswith('.pth')}

        rows = []
        common_files = golden_files & my_files
        if len(common_files) == 0:
            raise ValueError("Please ensure that both --my-path and --golden-path "
                             "have same name files.")
        for filename in common_files:
            logger.info("Start processing file: %r" % filename)
            key, token_id, parse_err = parse_key_id(filename)
            if parse_err:
                status = f"parse_error:{parse_err}"
                logger.warning(f"There is a invalid file, parse_error: {parse_err}")
                rows.append(create_rows(RowData(file_name=filename, cmp_fail_reason=status)))
            else:
                logger.info("Load file: %r , calculate cosine_similarity kl_divergence l1_norm ulp" % filename)
                golden_path = load_file_to_read_common_check(os.path.join(self.golden_path, filename))
                my_path = load_file_to_read_common_check(os.path.join(self.my_path, filename))
                golden_data = load_logits_from_file(golden_path)
                my_data = load_logits_from_file(my_path)
                compare_result = compare_tensor(golden_data, my_data, self.dtype)
                if compare_result[CMP_FAIL_REASON]:
                    rows.append(create_rows(RowData(file_name=filename, key=key, token_id=token_id, 
                                            cmp_fail_reason=compare_result[CMP_FAIL_REASON])))
                else:
                    rows.append(create_rows(RowData(file_name=filename, key=key, token_id=token_id,
                                            cosine_sim=compare_result[COS_SIMILARITY],
                                            kl_div=compare_result[KL_DIVERGENCE],
                                            l1_res=compare_result[L1_NORM],
                                            ulp_diff=compare_result[ULP_MAX_DIFF],
                                            ulp=compare_result[ULP],
                                            passed=None, cmp_fail_reason=None)))
        for filename in golden_files - my_files:
            rows.append(create_rows(RowData(file_name=filename, cmp_fail_reason="only_in_golden_path")))
        for filename in my_files - golden_files:
            rows.append(create_rows(RowData(file_name=filename, cmp_fail_reason="only_in_my_path")))
        return rows
    
    def compare_with_baseline(self, compare_result):
        logger.info(f"Compare with the baseline, cosine_similarity: {self.cosine_similarity},"
                    f"kl_divergence: {self.kl_divergence}, l1_norm: {self.l1_norm}")
        for result in compare_result:
            if len(result) != CMP_ROWS_LEN:
                raise ValueError("Error occur on compare result with baseline: the result format not correct")
            if result[CMP_FAIL_REASON_INDEX]:
                continue
            else:
                logger.debug(f"Compare result of cosine_similarity: {result[COS_SIMILARITY_INDEX]}."
                             f"Baseline: {self.cosine_similarity}")
                logger.debug(f"Compare result of kl_divergence: {result[KL_DIVERGENCE_INDEX]}."
                             f"Baseline: {self.kl_divergence}")
                logger.debug(f"Compare result of l1_norm: {result[L1_NORM_INDEX]}."
                             f"Baseline: {self.l1_norm}")
                if result[COS_SIMILARITY_INDEX] > self.cosine_similarity and \
                   result[KL_DIVERGENCE_INDEX] < self.kl_divergence and \
                   result[L1_NORM_INDEX] < self.l1_norm:
                    result[PASSED_INDEX] = "True"
                elif result[ULP_MAX_DIFF_INDEX] is not None and result[ULP_MAX_DIFF_INDEX] <= result[ULP_INDEX]:
                    result[PASSED_INDEX] = "True"
                else:
                    result[PASSED_INDEX] = "False"

    def save_result(self, rows):
        result_filename = "logits_cmp_res_" + get_timestamp() + ".csv"
        ms_makedirs(self.output_dir, mode=0o700, exist_ok=True)
        output_csv = os.path.join(self.output_dir, result_filename)
        logger.info("Save compare result to %r" % output_csv)
        with ms_open(output_csv, 'w', max_size=CSV_FILE_MAX_SIZE) as f:
            writer = csv.writer(f)
            writer.writerow([FILE_NAME, KEY, TOKEN_ID, COS_SIMILARITY, KL_DIVERGENCE, L1_NORM, ULP_MAX_DIFF, ULP,
                             PASSED, CMP_FAIL_REASON])
            for r in rows:
                if len(r) != CMP_ROWS_LEN:
                    logger.warning("There is an error row:" + " ".join(r))
                    continue
                formatted = [sanitize_csv_value(r[0])] + [r[i] for i in range(1, CMP_ROWS_LEN)]
                writer.writerow(formatted)

    def process_comparsion(self):
        rows = self.compare_logits()
        self.compare_with_baseline(rows)
        rows.sort(key=sort_key)
        self.save_result(rows)
        logger.info("Success compare logits of 'golden data' and 'my data'")
