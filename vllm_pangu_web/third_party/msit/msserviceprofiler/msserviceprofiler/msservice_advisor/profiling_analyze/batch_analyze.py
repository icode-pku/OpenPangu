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

import random
import numpy as np

from msserviceprofiler.msservice_advisor.profiling_analyze.register import register_analyze, cached, answer
from msserviceprofiler.msservice_advisor.profiling_analyze.register import GLOBAL_DATA
from msserviceprofiler.msservice_advisor.profiling_analyze.utils import TARGETS, SUGGESTION_TYPES, logger
from msserviceprofiler.msservice_advisor.profiling_analyze.utils import get_dict_value_by_pos, UmaskWrapper

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warning(f"Failed to import matplotlib.pyplot, cannot create a fit curve plot: {e}")
    plt = None


def summary_batch_info(batch_info):
    summary = {}
    for batchsize, latency_list in batch_info.items():
        if not latency_list or not isinstance(latency_list, list):
            continue  # 跳过空列表
        summary[batchsize] = {}
        latency_list.sort()

        summary[batchsize]["BSZ"] = batchsize
        summary[batchsize]["MIN"] = latency_list[0]
        summary[batchsize]["P10"] = latency_list[int(len(latency_list) * 0.1)]
        summary[batchsize]["P30"] = latency_list[int(len(latency_list) * 0.3)]
        summary[batchsize]["P50"] = latency_list[int(len(latency_list) * 0.5)]
        summary[batchsize]["P70"] = latency_list[int(len(latency_list) * 0.7)]
        summary[batchsize]["P90"] = latency_list[int(len(latency_list) * 0.9)]
        summary[batchsize]["MAX"] = latency_list[-1]
        summary[batchsize]["FIT_DATA"] = latency_list[int(len(latency_list) * 0.3): int(len(latency_list) * 0.7)]
    return summary


def print_list(array):
    for item in array:
        logger.debug(item)


@cached()
def read_batch_and_latency(pre_request):
    # 1. 获取所有的batchsize 对应的 latency
    # 2. 计算P50
    # 3. 打印出来
    prefill_batch_info = {}  # batchsize => list of latency
    decode_batch_info = {}  # batchsize => list of latency
    for request in pre_request.values():
        if not isinstance(request, dict):
            continue

        prefill_batch_size = request.get("prefill_bsz")
        decode_batch_size_list = request.get("decode_bsz", [])
        latency_list = request.get("latency", [])
        for index, latency in enumerate(latency_list):
            if index == 0:
                prefill_batch_info.setdefault(prefill_batch_size, [])
                prefill_batch_info[prefill_batch_size].append(latency)
            elif len(decode_batch_size_list) < index:
                logger.debug("Check the result_perf_*.csv data, decode_bsz not matching with latency")
                break
            else:
                decode_batch_size = decode_batch_size_list[index - 1]
                decode_batch_info.setdefault(decode_batch_size, [])
                decode_batch_info[decode_batch_size].append(latency)

    prefill_summary = summary_batch_info(prefill_batch_info)  # batchsize => P50
    decode_summary = summary_batch_info(decode_batch_info)  # batchsize => P50

    return prefill_summary, decode_summary


def find_best_by_curve_fit(summary_fit_data, process_name):
    from scipy.optimize import curve_fit, minimize

    max_batch_size = summary_fit_data[-1]["BSZ"]
    logger.debug(f"{process_name} 上次运行组的最大的 batch size 为 {max_batch_size}")

    if len(summary_fit_data) > 2:

        def func_curv(x, a, b, c):
            return a * x**b * np.exp(-c / x)  # 已检查 points 中不包含 0 值

    else:

        def func_curv(x, a, b):
            return a * x + b

    points = []
    targets = []

    for x in summary_fit_data:
        bsz = x["BSZ"]
        for latency in x["FIT_DATA"]:
            if latency == 0:
                # 如果 latency 为 0，直接抛出 ValueError 异常
                raise ValueError(f"latency 不能为 0，当前数据点为 (BSZ={bsz}, latency={latency})")
            if bsz == 0:
                # 如果 bsz 为 0，直接抛出 ValueError 异常
                raise ValueError(f"bsz 不能为 0，当前数据点为 (BSZ={bsz}, latency={latency})")
            points.append(bsz)
            targets.append(bsz * 1000 / latency)

    points.append(max_batch_size * 10)
    targets.append(0.00001)

    try:
        popt, pcov = curve_fit(func_curv, points, targets, maxfev=10000)
        logger.debug(f"{process_name} 函数拟合后参数：{popt}")

        # 或者使用数值优化（通用方法，适用于任何模型）
        def negative_func(x):
            return -func_curv(x, *popt)  # 最小化负函数即最大化原函数

        best_predicted = minimize(negative_func, x0=max_batch_size, bounds=[(0, max_batch_size * 2)])
        aggressive_predicted = minimize(negative_func, x0=max_batch_size, bounds=[(0, max_batch_size * 5)])
        logger.debug(f"{process_name} 搜索范围 2 倍当前最大batchsize. 结果是: {best_predicted.x[0]} {best_predicted}")
        logger.debug(
            f"{process_name} 搜索范围 5 倍当前最大batchsize. 结果是:  {aggressive_predicted.x[0]} {aggressive_predicted}"
        )

        result = {
            "best_batch_size": int(best_predicted.x[0]),
            "max_batch_size": max_batch_size,
            "points": points,
            "targets": targets,
            "popt": popt,
            "process_name": process_name,
            "func_curv": func_curv
        }
    except Exception as error:
        logger.warning(f"{process_name} 拟合失败：{error}")
        return {}

    return result


def get_predict_image(results):
    import datetime
    # 获取当前时间戳
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H%M%S")
    
    if plt is None:
        return

    len_result = len(results)
    if len_result <= 3:
        fig, axes = plt.subplots(1, len_result, figsize=(10 * len_result, 6))
        axes = [axes] if not isinstance(axes, (np.ndarray, list)) else axes
    else:
        num_rows = len_result // 3 + (len_result % 3 > 0)
        fig, axes = plt.subplots(num_rows, 3, figsize=(10 * 3, 6 * num_rows))
        axes = axes.flatten()
    for result, ax in zip(results, axes):
        max_batch_size = result.get('max_batch_size', 0)
        points = result.get('points', [])
        targets = result.get('targets', [])
        popt = result.get('popt', None)
        process_name = result.get('process_name', '')
        func_curv = result.get('func_curv', None)
        if func_curv is None or max_batch_size == 0:
            logger.info("func_curv is None")
            return
        # 开始画图
        x_values = np.linspace(1, max_batch_size * 5, 300)

        # 绘制模型预测均值和置信区间
        ax.plot(x_values, func_curv(x_values, *popt), label=f"Model Prediction", color="blue")
        ax.scatter(points[:-1], targets[:-1], c="green", s=100, edgecolor="black", label="Existing Data Points")

        ax.set_title(f"Curve Fit Optimization({process_name})")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Speed")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()

    png_name = f"func_curv_{timestamp}.png"
    logger.info(f"拟合画图路径：{png_name}")
    with UmaskWrapper(umask=0o137):
        plt.savefig(png_name)


def divide_fit_and_print(summary):
    if len(summary) == 0:
        return [], 0, []

    summary.sort(key=lambda x: x["BSZ"])
    best_bs = max(
        summary, key=lambda xx: (sum(xx["FIT_DATA"]) / len(xx["FIT_DATA"])) if len(xx["FIT_DATA"]) > 0 else 0
    )["BSZ"]
    to_fit = [dict(BSZ=x["BSZ"], FIT_DATA=x.pop("FIT_DATA")) for x in summary]
    return to_fit, best_bs, summary


def print_log_info(decode_to_print, prefill_to_print, prefill_cur_best_bs, decode_cur_best_bs):
    logger.debug("==decode==")
    print_list(decode_to_print)
    logger.debug("==prefill==")
    print_list(prefill_to_print)
    logger.debug(
        f"Best value from instance: prefill_cur_best_bs={prefill_cur_best_bs}, decode_cur_best_bs={decode_cur_best_bs}"
    )


def write_to_answer(suggest_item, action_item, reson_item):
    answer(
        suggesion_type=SUGGESTION_TYPES.config,
        suggesion_item=suggest_item,
        action=action_item,
        reason=reson_item
    )


def advise_according_len(decode_len, prefill_len):
    if decode_len <= 1:
        write_to_answer("maxBatchSize", "无法给出建议数据",
            f"decode batch 样本量 {decode_len} 太小，需要增大 batch size 重新获取 instance 数据")
    if prefill_len <= 1:
        write_to_answer("maxPrefillBatchSize", "无法给出建议数据",
            f"prefill batch 样本量 {prefill_len} 太小，需要增大 batch size 重新获取 instance 数据")


def generate_prefill_suggest_value(best_prefill_result, prefill_cur_best_bs):
    return max(best_prefill_result.get('best_batch_size', -1), prefill_cur_best_bs)


def generate_decode_suggest_value(best_decode_result, best_prefill_result, decode_cur_best_bs):
    suggest_value = max(
        best_decode_result.get('best_batch_size', -1),
        best_prefill_result.get('best_batch_size', -1),
        decode_cur_best_bs
    )
    max_batch_size_by_npu_mem = GLOBAL_DATA.get("maxBatchSize", {})
    if "min" in max_batch_size_by_npu_mem:
        suggest_value = max(suggest_value, max_batch_size_by_npu_mem["min"])
    if "max" in max_batch_size_by_npu_mem:
        suggest_value = min(suggest_value, max_batch_size_by_npu_mem["max"])
    return suggest_value


@register_analyze()
def find_best_batch_size(mindie_service_config, benchmark, profiling_params):
    if "results_per_request" not in benchmark:
        return

    results = []
    prefill_summary, decode_summary = read_batch_and_latency(benchmark.get("results_per_request", {}))
    decode_len, prefill_len = len(prefill_summary), len(decode_summary)
    if decode_len == 0 and prefill_len == 0:
        logger.warning(f"instance data not available, decode_len={decode_len}, prefill_len={prefill_len}")
        return

    np.seterr(invalid="ignore", divide="ignore", over="ignore")  # ignore all warning printing
    prefill_to_fit, prefill_cur_best_bs, prefill_to_print = divide_fit_and_print(list(prefill_summary.values()))
    decode_to_fit, decode_cur_best_bs, decode_to_print = divide_fit_and_print(list(decode_summary.values()))
    print_log_info(decode_to_print, prefill_to_print, prefill_cur_best_bs, decode_cur_best_bs)

    advise_according_len(decode_len, prefill_len)

    prefill_cur_config = get_dict_value_by_pos(
        mindie_service_config, "BackendConfig:ScheduleConfig:maxPrefillBatchSize"
    )
    decode_cur_config = get_dict_value_by_pos(mindie_service_config, "BackendConfig:ScheduleConfig:maxBatchSize")
    logger.debug(f"Config value: prefill_cur_config={prefill_cur_config}, decode_cur_config={decode_cur_config}")

    if prefill_len > 1:
        best_prefill_result = find_best_by_curve_fit(prefill_to_fit, "prefill")
        suggest_value = generate_prefill_suggest_value(best_prefill_result, prefill_cur_best_bs)
        if (prefill_cur_config and suggest_value != prefill_cur_config) or suggest_value:
            results.append(best_prefill_result)
            write_to_answer("maxPrefillBatchSize", f"尝试设置为 {suggest_value}，原值{prefill_cur_config or '未获取到'}",
                "经过当前不同batch的时延数据，通过函数拟合分析，建议最优batchsize")
    else:
        best_prefill_result = {}

    if decode_len > 1:
        best_decode_result = find_best_by_curve_fit(decode_to_fit, "decode")
        suggest_value = generate_decode_suggest_value(best_decode_result, best_prefill_result, decode_cur_best_bs)

        if (decode_cur_config and suggest_value != decode_cur_config) or suggest_value:
            results.append(best_decode_result)
            write_to_answer("maxBatchSize", f"尝试设置为 {suggest_value}，原值{decode_cur_config or '未获取到'}",
                "经过当前不同batch的时延数据，通过函数拟合分析，建议最优batchsize")
    try:
        if len(results) == 0:
            return
        get_predict_image(results)
    except Exception as error:
        logger.warning(f"图像生成失败: {error}")
    finally:
        np.seterr(invalid="warn", divide="warn", over="warn")  # set back to warn
        plt.close('all')