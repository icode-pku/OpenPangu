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

from pathlib import Path
import math

import numpy as np
import pandas as pd
from ms_service_profiler.exporters.base import ExporterBase
from ms_service_profiler.utils.log import logger

from msserviceprofiler.msguard.security.io import mkdir_s, open_s
from ..common.csv_fields import RequestCSVFields, BatchCSVFields, ServiceCSVFields
from ..common.constants import US_PER_MS


METRIC_COLUMN = "Metric"


def is_contained_valid_iter_info(rid_list, token_id_list):
    """
    检查 rid_list 和 token_id_list 是否为有效且长度相等的 list 或 tuple
    限制输入类型为list或tuple
    """
    # 检查是否为 None
    if rid_list is None or token_id_list is None:
        return False

    # 检查类型是否为list或者tuple
    if not isinstance(rid_list, (list, tuple)):
        return False

    if not isinstance(token_id_list, (list, tuple)):
        return False

    return len(rid_list) == len(token_id_list)


def print_warning_log(log_name):
    if not ExporterSummary.get_err_log_flag(log_name):
        logger.warning(f"The '{log_name}' field info is missing, please check.")
        ExporterSummary.set_err_log_flag(log_name, True)


def process_each_record(req_map, batch_map, record):
    process_req_record(req_map, record)
    process_batch_record(batch_map, record)


def process_batch_record(batch_map, record):
    if "batch_type" not in record or "rid_list" not in record:
        return

    batch_type = record.get('batch_type')
    rid_tuple = str(record.get('rid_list'))
    batch_size = 0
    during_time = 0

    raw_batch_size = record.get('batch_size')
    if raw_batch_size and str(raw_batch_size).isdigit():
        batch_size = int(raw_batch_size)
    raw_during_time = record.get('during_time')
    if raw_during_time and str(raw_during_time).isdigit():
        during_time = float(raw_during_time) / US_PER_MS

    # 构建batch_map
    if batch_type == 'Prefill':
        prefill_key = f"prefill_{rid_tuple}"
        if prefill_key not in batch_map:
            batch_map[prefill_key] = {
                BatchCSVFields.PREFILL_BATCH_NUM: 0,
                BatchCSVFields.PREFILL_EXEC_TIME: 0.0
            }
        batch_map[prefill_key][BatchCSVFields.PREFILL_BATCH_NUM] = batch_size
        batch_map[prefill_key][BatchCSVFields.PREFILL_EXEC_TIME] += during_time

    if batch_type == 'Decode':
        decode_key = f"decode_{rid_tuple}"
        if decode_key not in batch_map:
            batch_map[decode_key] = {
                BatchCSVFields.DECODE_BATCH_NUM: 0,
                BatchCSVFields.DECODE_EXEC_TIME: 0.0
            }
        batch_map[decode_key][BatchCSVFields.DECODE_BATCH_NUM] = batch_size
        batch_map[decode_key][BatchCSVFields.DECODE_EXEC_TIME] += during_time


def is_valid_number(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def process_req_record(req_map, record):
    name = record.get('name')
    rid = record.get('rid')
    req_wait_status = record.get('WAITING+')
    req_pend_status = record.get('PENDING+')
    recv_token = record.get('recvTokenSize=')
    reply_token = record.get('replyTokenSize=')

    if rid is None or name is None:
        print_warning_log('rid or name')
        return

    # req_map中存在rid则保存，不存在则默认
    if rid not in req_map:
        req_map[rid] = {
            'httpReq_start': None,
            'httpRes_end': None,
            'token_id': {},
            'req_waiting_time': 0.0,
            'req_pending_time': 0.0,
            'is_complete': False,
            'generated_token_num': 0,
            'input_token_num': 0,
            'exec_time': 0.0,
            'first_token_latency': 0.0,
        }
    entry = req_map[rid]

    # 处理不同name的记录
    if name == 'httpReq':
        entry['httpReq_start'] = record.get('start_time')
        entry['token_id'] = entry.get('token_id', {})  # 确保token_id存在
        entry.setdefault('req_waiting_time', 0.0)
        entry.setdefault('req_pending_time', 0.0)
        entry['is_complete'] = False  # 重置完成状态
    elif name == 'httpRes':
        entry['httpRes_end'] = record.get('end_time')
        entry['is_complete'] = True

    # 处理队列状态时间
    if req_wait_status == 1:
        entry['req_waiting_time'] = record.get('during_time', 0.0)
    if req_pend_status == 1:
        entry['req_pending_time'] = record.get('during_time', 0.0)

    # 处理Token数量
    if is_valid_number(recv_token):
        entry['input_token_num'] = recv_token
    if is_valid_number(reply_token):
        entry['generated_token_num'] = reply_token

    # 处理rid_list和token_id_list
    rid_list = record.get('rid_list')
    token_id_list = record.get('token_id_list')
    process_rid_token_list(req_map, rid_list, token_id_list, record)

    # 检查并移除空数据
    if is_empty_entry(entry):
        req_map.pop(rid)


def is_empty_entry(entry):
    """
    检查entry是否为空数据，即为默认值的场景
    """
    return (
        entry['httpReq_start'] is None
        and entry['httpRes_end'] is None
        and not entry['token_id']  # 空字典
        and entry['req_waiting_time'] == 0.0
        and entry['req_pending_time'] == 0.0
        and entry['is_complete'] is False
        and entry['generated_token_num'] is None
        and entry['input_token_num'] is None
    )


def process_rid_token_list(req_map, rid_list, token_id_list, record):
    if not is_contained_valid_iter_info(rid_list, token_id_list):
        return

    for i, value in enumerate(rid_list):
        req_rid = str(value)
        if req_map.get(req_rid) is None:
            print_warning_log('httpReq')
            continue

        # 执行总时长
        if record.get('name') == 'modelExec':
            during_time = record.get('during_time')
            if isinstance(during_time, (int, float)):
                req_map[req_rid]['exec_time'] += during_time
            else:
                logger.warning(f"Invalid during_time: {during_time} for rid={req_rid}")

        cur_iter = token_id_list[i]
        if cur_iter is None:
            print_warning_log('token_id_list')
            continue

        req_map[req_rid]['token_id'][str(cur_iter)] = record.get('end_time')

        # 首Token时延
        if cur_iter == 0:
            if req_map[req_rid].get('first_token_latency') is None:
                req_map[req_rid]['first_token_latency'] = record.get('during_time', 0)
            else:
                req_map[req_rid]['first_token_latency'] += record.get('during_time', 0)


def gen_exporter_results(all_data_df):
    req_map = {}
    batch_map = {}

    for _, record in all_data_df.iterrows():
        process_each_record(req_map, batch_map, record)

    # 生成request维度数据
    req_view, total_map = calculate_request_metrics(req_map)

    # 获取req_view数据
    first_token_latency = [req["first_token_latency"] for req in req_view]
    subsequent_token_latency = [
        latency
        for req in req_view
        for latency in req["subsequent_token_latency"]
    ]
    total_time = [req["total_time"] for req in req_view]
    exec_time = [req["exec_time"] for req in req_view]
    waiting_time = [req["waiting_time"] for req in req_view]
    input_token_num = [req["input_token_num"] for req in req_view]
    generated_token_num = [req["generated_token_num"] for req in req_view]

    # 计算统计值
    req_stats = {
        RequestCSVFields.FIRST_TOKEN_LATENCY: calculate_statistics(first_token_latency),
        RequestCSVFields.SUBSEQUENT_TOKEN_LATENCY: calculate_statistics(subsequent_token_latency),
        RequestCSVFields.TOTAL_TIME: calculate_statistics(total_time),
        RequestCSVFields.EXEC_TIME: calculate_statistics(exec_time),
        RequestCSVFields.WAITING_TIME: calculate_statistics(waiting_time),
        RequestCSVFields.INPUT_TOKEN_NUM: calculate_statistics(input_token_num),
        RequestCSVFields.GENERATED_TOKEN_NUM: calculate_statistics(generated_token_num)
    }

    # 生成batch维度数据
    batch_status = calculate_batch_metrics(batch_map)

    return req_stats, batch_status, total_map


def calculate_statistics(metric):
    """
    计算avg、max、min、p99、p90、p50的公共函数
    """
    if not metric or any(not isinstance(value, (int, float)) for value in metric):
        return {
            "avg": np.nan,
            "max": np.nan,
            "min": np.nan,
            "p50": np.nan,
            "p90": np.nan,
            "p99": np.nan
        }
    return {
        "avg": round(np.mean(metric), 4),
        "max": round(np.max(metric), 4),
        "min": round(np.min(metric), 4),
        "p50": round(np.percentile(metric, 50), 4),
        "p90": round(np.percentile(metric, 90), 4),
        "p99": round(np.percentile(metric, 99), 4)
    }


def calculate_batch_metrics(batch_map):
    prefill_batch_num_list = []
    decode_batch_num_list = []
    prefill_exec_time_list = []
    decode_exec_time_list = []

    for key, value in batch_map.items():
        if key.startswith('prefill'):
            prefill_batch_num_list.append(value.get(BatchCSVFields.PREFILL_BATCH_NUM, 0))
            prefill_exec_time_list.append(value.get(BatchCSVFields.PREFILL_EXEC_TIME, 0.0))
        elif key.startswith('decode'):
            decode_batch_num_list.append(value.get(BatchCSVFields.DECODE_BATCH_NUM, 0))
            decode_exec_time_list.append(value.get(BatchCSVFields.DECODE_EXEC_TIME, 0.0))

    # 计算统计指标
    batch_status = {
        BatchCSVFields.PREFILL_BATCH_NUM: calculate_statistics(prefill_batch_num_list),
        BatchCSVFields.DECODE_BATCH_NUM: calculate_statistics(decode_batch_num_list),
        BatchCSVFields.PREFILL_EXEC_TIME: calculate_statistics(prefill_exec_time_list),
        BatchCSVFields.DECODE_EXEC_TIME: calculate_statistics(decode_exec_time_list)
    }

    return batch_status


def get_non_first_token_latency(req_data):
    token_id = req_data["token_id"]
    subsequent_token_latency = []
    sorted_tokens = sorted(token_id.items(), key=lambda x: int(x[0]))
    for i in range(1, len(sorted_tokens)):
        current_token_time = sorted_tokens[i][1]
        previous_token_time = sorted_tokens[i - 1][1]
        latency = round((current_token_time - previous_token_time) / US_PER_MS, 4)
        subsequent_token_latency.append(latency)
    return subsequent_token_latency


def gen_result_record(req_id, req_data):
    input_token_num = req_data["input_token_num"]
    generated_token_num = req_data["generated_token_num"]

    # 从字典中取出数据
    record = {
        "req_id": req_id,
        "first_token_latency": round(req_data["first_token_latency"] / US_PER_MS, 4),
        "exec_time": round(req_data["exec_time"] / US_PER_MS, 4),
        "input_token_num": input_token_num,
        "generated_token_num": generated_token_num
    }

    # 检查 httpReq_start 和 httpRes_end 的有效性
    if req_data["httpReq_start"] is not None and req_data["httpRes_end"] is not None:
        total_time = (req_data["httpRes_end"] - req_data["httpReq_start"]) / US_PER_MS
        record["total_time"] = round(total_time, 4)
    else:
        record["total_time"] = 0.0  # 无效数据

    # 非首Token时延
    record["subsequent_token_latency"] = get_non_first_token_latency(req_data)

    # 队列等待时长
    waiting_time = req_data["req_pending_time"] + req_data["req_waiting_time"]
    record["waiting_time"] = waiting_time
    return record


def calculate_request_metrics(req_map):
    req_view = []
    total_map = {
        ServiceCSVFields.TOTAL_INPUT_TOKEN_NUM: 0,
        ServiceCSVFields.TOTAL_GENERATED_TOKEN_NUM: 0,
        ServiceCSVFields.GENERATE_TOKEN_SPEED: 0,
        ServiceCSVFields.GENERATE_ALL_TOKEN_SPEED: 0
    }

    valid_requests = []
    first_request_start_time, last_request_end_time = None, None

    # 过滤无效请求时同时保存 req_id
    for req_id, req_data in req_map.items():
        if req_data["httpReq_start"] is not None and req_data["httpRes_end"] is not None:
            valid_requests.append((req_id, req_data))

    # 处理有效请求时解包 req_id 和 req_data
    for req_id, req_data in valid_requests:
        record = gen_result_record(req_id, req_data)
        req_view.append(record)

        # 更新总体维度数据
        total_map[ServiceCSVFields.TOTAL_INPUT_TOKEN_NUM] += req_data.get("input_token_num", 0)
        total_map[ServiceCSVFields.TOTAL_GENERATED_TOKEN_NUM] += req_data.get("generated_token_num", 0)

        # 更新第一个和最后一个请求的时间
        current_start_time = req_data["httpReq_start"]
        current_end_time = req_data["httpRes_end"]
        if first_request_start_time is None or current_start_time < first_request_start_time:
            first_request_start_time = current_start_time
        if last_request_end_time is None or current_end_time > last_request_end_time:
            last_request_end_time = current_end_time

    total_exec_time = 0.0

    # 计算 total_exec_time
    if first_request_start_time is not None and last_request_end_time is not None:
        total_exec_time = round((last_request_end_time - first_request_start_time) / 1000000, 4)

    if total_exec_time > 0:
        total_map[ServiceCSVFields.GENERATE_TOKEN_SPEED] = round(
            total_map[ServiceCSVFields.TOTAL_GENERATED_TOKEN_NUM] / total_exec_time, 4
        )
        total_map[ServiceCSVFields.GENERATE_ALL_TOKEN_SPEED] = round(
            (total_map[ServiceCSVFields.TOTAL_INPUT_TOKEN_NUM] +
             total_map[ServiceCSVFields.TOTAL_GENERATED_TOKEN_NUM]) / total_exec_time, 4
        )

    return req_view, total_map


def save_dataframe_to_csv(map_data, output, file_name, include_stats=1):
    if output is None:
        return

    output_path = Path(output)
    mkdir_s(output)
    file_path = output_path / file_name

    # 将map_data转换为DataFrame
    df = convert_map_to_dataframe(map_data, include_stats)

    # 处理数据转换
    value_columns = df.columns.drop(METRIC_COLUMN, errors="ignore")
    df[value_columns] = df[value_columns].fillna(0)
    for col in value_columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    # 写入CSV文件
    try:
        with open_s(file_path, "w") as f:
            df.to_csv(f, index=False)
            logger.info(f"Write to {file_name} success.")
    except Exception as e:
        logger.warning(f"Failed to write to {file_name}, {e}")


def convert_map_to_dataframe(map_data, include_stats):
    if isinstance(map_data, pd.DataFrame):
        return map_data
    data = []
    for metric, values in map_data.items():
        if include_stats == 1:
            row = {
                BatchCSVFields.METRIC: metric,
                BatchCSVFields.AVG: values["avg"],
                BatchCSVFields.MAX: values["max"],
                BatchCSVFields.MIN: values["min"],
                BatchCSVFields.P50: values["p50"],
                BatchCSVFields.P90: values["p90"],
                BatchCSVFields.P99: values["p99"]
            }
        else:
            value = format(values, ".8f") if metric == ServiceCSVFields.GENERATE_TOKEN_SPEED else values
            row = {ServiceCSVFields.METRIC: metric, ServiceCSVFields.VALUE: value}
        data.append(row)
    return pd.DataFrame(data)


def drop_all_zero_rows(status, drop_rows, include_stats=1):
    status = convert_map_to_dataframe(status, include_stats)
    # 筛选出 Metric 列值在 drop_rows 中的行
    mask_in_drop_rows = status[METRIC_COLUMN].isin(drop_rows)
    value_columns = status.columns.drop(METRIC_COLUMN)
    mask_all_zero = (status[value_columns] == 0).all(axis=1)
    # 综合条件：在 drop_rows 中 且 其他列全为 0 → 删除
    status = status.loc[~(mask_in_drop_rows & mask_all_zero)]
    return status


def get_new_ttft_wait_time(data):
    ttft_df = data.get('req_ttft_df', pd.DataFrame())
    que_wait_df = data.get('req_que_wait_df', pd.DataFrame())
    if ttft_df.empty or que_wait_df.empty:
        return {}
    ttft_df.loc[:, 'ttft'] = ttft_df['ttft'].div(US_PER_MS)
    que_wait_df.loc[:, 'que_wait_time'] = que_wait_df['que_wait_time'].div(US_PER_MS)
    ttdf_list = ttft_df['ttft'].fillna(0).to_list() 
    que_wait_list = que_wait_df['que_wait_time'].fillna(0).to_list()
    return {
        RequestCSVFields.FIRST_TOKEN_LATENCY: calculate_statistics(ttdf_list),
        RequestCSVFields.WAITING_TIME: calculate_statistics(que_wait_list)
    }


def is_invaild_rid(rid):
    return ',' in rid or '{' in rid or ':' in rid


def get_new_total_time(all_data_df):
    req_group_df = all_data_df.groupby('rid')
    total_times = []
    for rid, pre_req_data in req_group_df:
        rid = str(rid)
        if rid == '' or is_invaild_rid(rid):
            continue
        start_time = -1
        end_time = -1

        # 获取httpReq
        http_req_df = pre_req_data[pre_req_data['name'] == 'httpReq']
        if not http_req_df.empty:
            first_row = http_req_df.iloc[0]
            start_time = first_row.get('start_time', 0)

        # 获取 httpRes
        # 由于存在httpRes提前被调用，导致请求结束时间过早的情况，所以当前取httpRes和DecodeEnd中最晚一个点作为请求结束时间
        # mindIE重构后，取最后一个sendResponse的结束时间
        http_res_df = pre_req_data[pre_req_data['name'].isin(['httpRes', 'DecodeEnd', 'sendResponse'])]
        if not http_res_df.empty:
            last_row = http_res_df.iloc[-1]
            end_time = last_row.get("end_time", 0)

        # 计算 execution_time
        if start_time != -1 and end_time != -1 and end_time > start_time:
            total_time = (end_time - start_time) / US_PER_MS
            total_times.append(total_time)
    
    if not total_times:
        return {}
    return {RequestCSVFields.TOTAL_TIME: calculate_statistics(total_times)}


class ExporterSummary(ExporterBase):
    name = "summary"
    err_log = {'rid or name': False, 'start_time': False, 'httpReq': False, 'token_id_list': False}

    @classmethod
    def initialize(cls, args):
        cls.args = args
        cls.err_log = {'rid or name': False, 'start_time': False, 'httpReq': False, 'token_id_list': False}

    @classmethod
    def set_err_log_flag(cls, index, value):
        cls.err_log[index] = value

    @classmethod
    def get_err_log_flag(cls, index):
        return cls.err_log[index]

    @classmethod
    def export(cls, data) -> None:
        all_data_df = data.get('tx_data_df')
        output = cls.args.output_path

        if all_data_df is None:
            logger.warning("The data is empty, please check")
            return

        all_data_df = all_data_df[all_data_df['domain'] != 'KVCache']

        # 调用计算首Token时延的函数
        req_status, batch_status, total_map = gen_exporter_results(all_data_df)
        
        total_time_dict = get_new_total_time(all_data_df)
        if total_time_dict:
            req_status.update(total_time_dict)
        ttft_wait_time = get_new_ttft_wait_time(data)
        if ttft_wait_time:
            req_status.update(ttft_wait_time)

        batch_status = drop_all_zero_rows(batch_status, 
                                          [BatchCSVFields.PREFILL_EXEC_TIME, BatchCSVFields.DECODE_EXEC_TIME])

        req_status = drop_all_zero_rows(req_status, [RequestCSVFields.EXEC_TIME])

        # 格式化存入csv
        save_dataframe_to_csv(req_status, output, RequestCSVFields.PATH_NAME)
        save_dataframe_to_csv(batch_status, output, BatchCSVFields.PATH_NAME)
        save_dataframe_to_csv(total_map, output, ServiceCSVFields.PATH_NAME, include_stats=0)