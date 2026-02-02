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

import argparse
from textwrap import dedent

from .base import CommandType
from .legacy import add_legacy_argument
from ..utils import ErrorSeverity


def setup_precheck_parser(subparsers):
    desc = dedent('''\
        PRECHECK - Run a seires of validations and checks for different PD deployment.

        Mix Mode (default):
          - Validate environment, system, and HCCL settings.
          - Generate 'msprechecker_env.sh' if environment issues found.

        Disaggregation Mode:
          - Validate only user_config.json and mindie_env.json.
          - Skip environment/system/HCCL checks.
    ''')
    epilog = dedent('''\
        Examples:
          msprechecker precheck --rank-table hccl_8s_64p.json --weight-dir ./Llama3-70B
              # Full validation (default)
          msprechecker precheck --user-config-path /config/user.json --mindie-env-path /env/mindie.json
              # Disaggregation mode
    ''')

    precheck_parser = subparsers.add_parser(
        CommandType.CMD_PRECHECK.value,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=desc, usage='msprechecker precheck [OPTIONS]',
        epilog=epilog,
        help="Run comprehensive system validation for different PD deployment scenarios"
    )

    add_legacy_argument(precheck_parser)
    _add_pd_disagg_args(precheck_parser)
    _add_pd_mix_args(precheck_parser)
    _add_network_args(precheck_parser)
    _add_model_args(precheck_parser)
    _add_stress_test_args(precheck_parser)
    _add_custom_validation_args(precheck_parser)

    return precheck_parser


def _add_pd_disagg_args(parser):
    group = parser.add_argument_group("PD Disaggregation Options")
    group.add_argument(
        "--scene", metavar="",
        help="Specify different deploy scene. Supports: pd_disaggregation, " \
        "pd_disaggregation_single_container, mindie, vllm, vllm,ep."
    )
    group.add_argument(
        "--user-config-path", metavar="",
        help="Path to the 'user_config.json' file for Kubernetes-based deployments."
    )
    group.add_argument(
        "--mindie-env-path", metavar="",
        help="Path to the 'mindie_env.json' file for Kubernetes-based deployments."
    )
    group.add_argument(
        "--config-parent-dir", metavar="",
        help="Path to the parent directory for Kubernetes-based deployments"
    )


def _add_pd_mix_args(parser):
    group = parser.add_argument_group("PD Mixed Mode Options")
    group.add_argument(
        "--mies-config-path",
        metavar="", help="Path to the 'config.json' file for daemon-based deployments."
    )


def _add_network_args(parser):
    group = parser.add_argument_group("Network Options")
    group.add_argument(
        "--rank-table-path",
        metavar="", help="Path to the rank table file. Supports both A2 and A3 formats."
    )


def _add_model_args(parser):
    group = parser.add_argument_group("Model Options")
    group.add_argument(
        "--weight-dir", metavar="",
        help="Directory path containing model weights."
    )


def _add_stress_test_args(parser):
    group = parser.add_argument_group("Stress Test Options")
    group.add_argument(
        "--hardware", action="store_true",
        default=False, help="Enable hardware stress testing. Default: False."
    )
    group.add_argument(
        "--threshold", type=int,
        choices=range(0, 101), default=20,
        metavar="0-100", help="Set the failure threshold percentage (0-100). Default: 20."
    )


def _add_custom_validation_args(parser):
    group = parser.add_argument_group("Custom Validation Options")
    group.add_argument(
        "--custom-config-path", metavar="",
        help="Path to a custom validation rules configuration file."
    )
    group.add_argument(
        "-l", "--severity-level", metavar="",
        choices=[
            ErrorSeverity.ERR_LOW,
            ErrorSeverity.ERR_MEDIUM,
            ErrorSeverity.ERR_HIGH
        ], default=ErrorSeverity.ERR_LOW,
        type=ErrorSeverity, 
        help="Report only issues with the specified severity level or higher. Default: low."
    )
