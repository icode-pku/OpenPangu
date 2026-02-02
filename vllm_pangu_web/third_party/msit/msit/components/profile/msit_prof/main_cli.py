# Copyright (c) 2023-2025 Huawei Technologies Co., Ltd.
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

import argparse
import re

from components.utils.parser import BaseCommand
from components.utils.security_check import check_output_path_legality, check_input_opsummary_legality, \
                                            valid_ops_map_file
from msit_prof.msprof.msprof_process import msprof_process
from msit_prof.msprof.args_adapter import MsProfArgsAdapter
from msit_prof.analyze.autofuse.single_op_analyze import SingleOpAnalyzer


def check_application_string_legality(value):
    cmd_str = value
    regex = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{;: /.~-]")
    if regex.search(cmd_str):
        raise argparse.ArgumentTypeError(f"application string \"{cmd_str}\" is not a legal string")
    return cmd_str


class ProfileCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "--application",
            type=check_application_string_legality,
            required=True,
            help="Configure to run AI task files on the environment"
        )
        parser.add_argument(
            "-o",
            "--output",
            type=check_output_path_legality,
            default=None,
            help="The storage path for the collected profiling data,"
                " which defaults to the directory where the app is located"
        )
        parser.add_argument(
            "--model-execution",
            default="on",
            choices=["on", "off"],
            help="Control ge model execution performance data collection switch"
        )
        parser.add_argument(
            "--sys-hardware-mem",
            default="on",
            choices=["on", "off"],
            help="Control the read/write bandwidth data acquisition switch for ddr and llc"
        )
        parser.add_argument(
            "--sys-cpu-profiling",
            default="off",
            choices=["on", "off"],
            help="CPU acquisition switch"
        )
        parser.add_argument(
            "--sys-profiling",
            default="off",
            choices=["on", "off"],
            help="System CPU usage and system memory acquisition switch"
        )
        parser.add_argument(
            "--sys-pid-profiling",
            default="off",
            choices=["on", "off"],
            help="The CPU usage of the process and the memory collection switch of the process"
        )
        parser.add_argument(
            "--dvpp-profiling",
            default="on",
            choices=["on", "off"],
            help="DVPP acquisition switch"
        )
        parser.add_argument(
            "--runtime-api",
            default="on",
            choices=["on", "off"],
            help="Control runtime api performance data collection switch"
        )
        parser.add_argument(
            "--task-time",
            default="on",
            choices=["on", "off"],
            help="Control ts timeline performance data collection switch"
        )
        parser.add_argument(
            "--aicpu",
            default="on",
            choices=["on", "off"],
            help="Control aicpu performance data collection switch"
        )

    def handle(self, args):
        args_adapter = MsProfArgsAdapter(args.application, args.output, args.model_execution, args.sys_hardware_mem,
                                         args.sys_cpu_profiling, args.sys_profiling, args.sys_pid_profiling,
                                         args.dvpp_profiling, args.runtime_api, args.task_time, args.aicpu)
        return msprof_process(args_adapter)


class AnalyzeCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "--mode",
            dest="mode",
            required=False,
            default="graph",
            choices=["graph"],
            help="Configure the model inference tuning scenario. Currently, only 'graph' mode are supported."
        )
        parser.add_argument(
            "-f",
            "--framework",
            dest="framework",
            required=False,
            default="tf",
            choices=["tf"],
            help="Specify the AI framework to use. Currently, only 'TensorFlow' is supported."
        )
        parser.add_argument(
            "--origin",
            dest="origin",
            type=check_input_opsummary_legality,
            required=True,
            help="Specify the path of op_summary data collected after disabling all fusion strategies."
        )
        parser.add_argument(
            "--fused",
            dest="fused",
            type=check_input_opsummary_legality,
            required=True,
            help="Specify the path of op_summary data collected after ebale 'autofuse' fusion strategies."
        )
        parser.add_argument(
            "-ops",
            "--ops-graph",
            dest="ops_graph",
            type=valid_ops_map_file,
            required=True,
            help="Specify the file path of dumped GE graph."
        )
        parser.add_argument(
            "-o",
            "--output",
            dest="output",
            type=check_output_path_legality,
            required=False,
            default="./",
            help="Specify the save path for the performance analysis result."
        )

    def handle(self, args):
        autofuse_analyzer = SingleOpAnalyzer(args)
        autofuse_analyzer.analyze()


def get_cmd_instance():
    profile_help_info = "Provides a one-stop performance tuning and analysis tools."
    analyze_help_info = "Supports the analysis of profiling data and provides analysis reports" \
                        "to guide model performance tuning."
    msprof_instance = ProfileCommand("msprof", "get profiling data of a given programma.")
    analyze_instance = AnalyzeCommand("analyze", analyze_help_info)
    instances = [
        msprof_instance, analyze_instance
    ]
    return BaseCommand("profile", profile_help_info, instances)