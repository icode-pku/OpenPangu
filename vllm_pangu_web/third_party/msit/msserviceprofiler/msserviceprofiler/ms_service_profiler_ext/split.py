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
import re

from msserviceprofiler.msguard import validate_args, Rule
from msserviceprofiler.msguard.security.io import mkdir_s


def add_exporters(args):
    from msserviceprofiler.ms_service_profiler_ext.exporters.exporter_prefill import ExporterPrefill
    from msserviceprofiler.ms_service_profiler_ext.exporters.exporter_decode import ExporterDecode

    if not hasattr(args, "format"):
        args.format = "csv"
    exporter_cls = []
    exporters = []
    if args.prefill_batch_size > 0 or args.prefill_rid != "-1":
        exporter_cls.append(ExporterPrefill)
    if args.decode_batch_size > 0 or args.decode_rid != "-1":
        exporter_cls.append(ExporterDecode)
    for cls in exporter_cls:
        exporter = cls()
        exporter.initialize(args)
        exporters.append(exporter)
    return exporters


def check_string_valid(s, max_length=256):
    if len(s) > max_length:
        raise argparse.ArgumentTypeError("String length exceeds %d characters: %r" % (max_length, s))
    if not re.match(r"^[a-zA-Z0-9_-]+$", s):
        raise argparse.ArgumentTypeError("Unsafe string: %r" % s)
    return s


def check_non_negative_integer(value):
    try:
        value = int(value)
    except Exception as e:
        raise ValueError(f"'{value}' cannot convert to a positive integer.") from e
    
    if value < 0:
        raise ValueError(f"'{value}' is not a positive integer.")
    
    return value


def arg_parse(subparsers):
    parser = subparsers.add_parser(
        "split", formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="MS Server Profiler Split"
    )
    parser.add_argument(
        "--input-path",
        required=True,
        type=validate_args(Rule.input_dir_traverse),
        help="Path to the folder containing profile data.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join(os.getcwd(), "output"),
        help="Output file path to save results.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "fatal", "critical"],
        help="Log level to print")
    
    prefill_group = parser.add_argument_group("Prefill Parameters")
    prefill_group.add_argument(
        "--prefill-batch-size", type=check_non_negative_integer, default=0, help="Batch size for Prefill data."
    )
    prefill_group.add_argument(
        "--prefill-number", type=check_non_negative_integer, 
        default=1, help="The number of Prefill batch to calc statistical data"
    )
    prefill_group.add_argument(
        "--prefill-rid", type=lambda x: check_string_valid(x, max_length=100),
        default="-1", help="The rid for Prefill batch to split"
    )

    # 创建Decode参数组
    decode_group = parser.add_argument_group("Decode Parameters")
    decode_group.add_argument(
        "--decode-batch-size", type=check_non_negative_integer, default=0, help="Batch size for Decode data."
    )
    decode_group.add_argument(
        "--decode-number", type=check_non_negative_integer, 
        default=1, help="The number of Decode batch to calc statistical data"
    )
    decode_group.add_argument(
        "--decode-rid", type=lambda x: check_string_valid(x, max_length=100),
        default="-1", help="The rid for Decode batch to split"
    )
    parser.set_defaults(func=main)


def main(args):
    from ms_service_profiler.parse import parse
    from ms_service_profiler.utils.log import set_log_level
    from ms_service_profiler.plugins import custom_plugins

    # 初始化日志等级
    set_log_level(args.log_level)

    # 初始化Exporter
    exporters = add_exporters(args)

    # 检查output目录
    mkdir_s(args.output_path)
    if not Rule.output_dir._is_satisfied_by(args.output_path):
        raise argparse.ArgumentTypeError(f"Output path is not valid: {args.output_path!r}")

    # 解析数据并导出
    parse(args.input_path, custom_plugins, exporters, args=args)
