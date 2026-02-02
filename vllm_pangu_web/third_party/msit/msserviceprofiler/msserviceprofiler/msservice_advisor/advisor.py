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

import os
import argparse
from collections import namedtuple
from dataclasses import dataclass

from msserviceprofiler.msservice_advisor.profiling_analyze.utils import TARGETS, LOG_LEVELS, SUGGESTION_TYPES
from msserviceprofiler.msservice_advisor.profiling_analyze.utils import str_ignore_case, logger, set_log_level
from msserviceprofiler.msservice_advisor.profiling_analyze.utils import get_latest_matching_file, read_csv_or_json
from msserviceprofiler.msservice_advisor.profiling_analyze.utils import is_root

from msserviceprofiler.msguard import validate_args, Rule


# 文件格式，字典："21559056a7ff44c88a891ecbb537c431"："0", ...
REQ_TO_DATA_MAP_PATTERN = "req_to_data_map.json"

# FirstTokenTime,DecodeTime,LastDecodeTime,...
# 213.2031 ms,228.3775 ms,88.327 ms,...
# -> average, max, min, P75, P90, SLO_P90, P99, N
RESULT_PERF_PATTERN = "result_perf_*.csv"
PERF_METRICS = ["average", "max", "min", "P75", "P90", "SLO_P90", "P99", "N"]
PERF_METRICS_MAP = {str_ignore_case(ii): ii for ii in PERF_METRICS}

# ...,Concurrency,ModelName,lpct,Throughput,GenerateSpeed,...
# ...,50,DeepSeek-R1,0.9336 ms,2.789 req/s,...
RESULT_COMMON_PATTERN = "result_common_*.csv"

# 文件格式，字典："7" -> "input_len"：213, "output_len"：12, "prefill_bsz"：15, "decode_bsz"：[20, ...],
# 文件格式，字典： "req_latency"：2348332643508911, "latency"：[798.7475395202637, ...], "queue_latency"：[445314, ...], ... 
RESULTS_PER_REQUEST_PATTERN = "results_per_request_*.json"

MIES_INSTALL_PATH = "MIES_INSTALL_PATH"
MINDIE_SERVICE_DEFAULT_PATH = "/usr/local/Ascend/mindie/latest/mindie-service"

TARGETS_MAP = {
    "ttft": TARGETS.FirstTokenTime,
    "firsttokentime": TARGETS.FirstTokenTime,
    "throughput": TARGETS.Throughput,
}

LOG_LEVELS_LOWER = [ii.lower() for ii in LOG_LEVELS.keys()]


@dataclass
class ProfilingParameters:
    target: str
    target_metrics: str
    input_token_num: int
    output_token_num: int
    tp: int

    @classmethod
    def extract_from_args(cls, args):
        return cls(
            target=args.target,
            target_metrics=args.target_metrics,
            input_token_num=args.input_token_num,
            output_token_num=args.output_token_num,
            tp=args.tp
        )


def get_next_dict_item(dict_value):
    return dict([next(iter(dict_value.items()))])


def parse_benchmark_instance(instance_path):
    if not instance_path:
        logger.warning(f"instance_path not provided or not accessible, will skip related analyse.")
        return {}

    logger.debug("\nreq_to_data_map:")
    req_to_data_map = read_csv_or_json(get_latest_matching_file(instance_path, REQ_TO_DATA_MAP_PATTERN))
    logger.debug(f"req_to_data_map: {get_next_dict_item(req_to_data_map) if req_to_data_map else None}")

    logger.debug("\nresult_perf:")
    result_perf = read_csv_or_json(get_latest_matching_file(instance_path, RESULT_PERF_PATTERN))
    result_perf = {kk: dict(zip(PERF_METRICS, vv)) for kk, vv in result_perf.items()} if result_perf else {}
    logger.debug(f"result_perf: {get_next_dict_item(result_perf) if result_perf else None}")

    logger.debug("\nresult_common:")
    result_common = read_csv_or_json(get_latest_matching_file(instance_path, RESULT_COMMON_PATTERN))
    logger.debug(f"result_common: {result_common if result_common else None}")

    logger.debug("\nresults_per_request:")
    results_per_request = read_csv_or_json(get_latest_matching_file(instance_path, RESULTS_PER_REQUEST_PATTERN))
    logger.debug(f"results_per_request: {get_next_dict_item(results_per_request) if results_per_request else None}")

    return dict(
        req_to_data_map=req_to_data_map if req_to_data_map else {},
        result_perf=result_perf if result_perf else {},
        result_common=result_common if result_common else {},
        results_per_request=results_per_request if results_per_request else {},
    )


def get_mindie_server_config_path(service_config_path):
    if not service_config_path:
        service_config_path = os.getenv(MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH)

    logger.debug("\nmindie_service_config:")
    if not service_config_path.endswith(".json"): # mindie service path
        service_config_path = os.path.join(service_config_path, "conf", "config.json")
    
    return service_config_path


def parse_mindie_server_config(service_config_path):
    service_config_path = get_mindie_server_config_path(service_config_path)

    if not Rule.input_file_read.is_satisfied_by(service_config_path):
        logger.warning(f"mindie_service_path not provided or not accessible, will skip related analyse.")
        return None

    logger.info(f"mindie_service_path: {service_config_path}")
    mindie_service_config = read_csv_or_json(service_config_path)

    logger.debug(
        f"mindie_service_config: {get_next_dict_item(mindie_service_config) if mindie_service_config else None}"
    )

    return mindie_service_config


def analyze(mindie_service_config, benchmark_instance, params: ProfilingParameters):
    import msserviceprofiler.msservice_advisor.profiling_analyze
    from msserviceprofiler.msservice_advisor.profiling_analyze.register import REGISTRY, ANSWERS

    logger.info("")
    logger.info("<think>")
    for name, analyzer in REGISTRY.items():
        logger.info(f"[{name}]")
        analyzer(mindie_service_config, benchmark_instance, params)
    logger.info("</think>")

    logger.info("")
    logger.info("<advice>")
    for suggesion_type in SUGGESTION_TYPES:
        for name, items in ANSWERS.get(suggesion_type, dict()).items():
            for action, reason in items:
                logger.info(f"[{suggesion_type}] {name}")
                logger.info(f"[advice] {action}")
                logger.info(f"[reason] {reason}")
                logger.info("")
    logger.info("</advice>")


def check_positive_integer(value):
    try:
        value = int(value)
    except Exception as e:
        raise ValueError(f"'{value}' cannot convert to a positive integer.") from e

    if value < 0:
        raise ValueError(f"'{value}' is not a positive integer.")

    return value


def arg_parse(subparsers):
    parser = subparsers.add_parser(
        "advisor", formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="advisor for MindIE Service performance"
    )
    parser.add_argument(
        "-i", "--instance_path", type=validate_args(Rule.input_dir_traverse), required=False,
        default=None, help="benchamrk instance output directory"
    )
    parser.add_argument(
        "-s", "--service_config_path", type=str, required=False,
        default=None, help="MindIE Service config json path"
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str_ignore_case,
        default="ttft",
        choices=list(TARGETS_MAP.keys()),
        help="profiling key target",
    )
    parser.add_argument(
        "-m",
        "--target_metrics",
        type=lambda xx: PERF_METRICS_MAP.get(str_ignore_case(xx), None),
        default="average",
        choices=PERF_METRICS,
        help="profiling key target metrics",
    )
    parser.add_argument(
        "-in", "--input_token_num", type=check_positive_integer, default=0, help="input token number"
    )
    parser.add_argument(
        "-out", "--output_token_num", type=check_positive_integer, default=0, help="output token number"
    )
    parser.add_argument("-tp", "--tp", type=check_positive_integer, default=0, help="tp")
    parser.add_argument("--log_level", "-l", default="info", choices=LOG_LEVELS_LOWER, help="specify log level.")
    parser.set_defaults(func=main)


def main(args):
    if is_root():
        logger.warning(
            "Security Warning: Do not run this tool as root. "
            "Running with elevated privileges may compromise system security. "
            "Use a regular user account."
        )
    profiling_params = ProfilingParameters.extract_from_args(args)
    set_log_level(args.log_level)
    benchmark_instance = parse_benchmark_instance(args.instance_path)
    mindie_service_config = parse_mindie_server_config(args.service_config_path)
    analyze(mindie_service_config, benchmark_instance, profiling_params)
