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

from .legacy import add_legacy_argument


def setup_dump_parser(subparsers):
    dump_parser = subparsers.add_parser(
        "dump",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_get_dump_description(),
        usage='msprechecker dump [EXTRA OPTIONS] [--output-path <PATH>]',
        epilog=_get_dump_epilog(),
        help="Dump the current context for later comparison",
    )

    add_legacy_argument(dump_parser, True)
    _add_dump_arguments(dump_parser)
    _add_extra_options(dump_parser)

    return dump_parser


def _get_dump_description():
    return dedent('''\
        DUMP - Collect and save the current environment, system, and configuration context.

        This command gathers environment variables, system information, configuration files,
        and network topology, then saves them to a specified output file for later comparison.
    ''')


def _get_dump_epilog():
    return dedent('''\
        Example:
          msprechecker dump                                                                           # Default saved to current dir 'msprechecker_dumped.json'
          msprechecker dump --output-path /output/path                                                # Save snapshots to custom path: '/output/path'
          msprechecker dump --user-config-path user_config.json --mindie-env-path mindie_env.json     # Dump extra PD disaggregation configuration files
    ''')


def _add_dump_arguments(dump_parser):
    dump_parser.add_argument(
        "--output-path",
        metavar="",
        default="./msprechecker_dumped.json",
        help=(
            "Path to save the dumped context (JSON format). "
            "Default: './msprechecker_dumped.json'."
        )
    )


def _add_extra_options(dump_parser):
    extra_group = dump_parser.add_argument_group("Extra Options")
    extra_group.add_argument(
        "--filter",
        action="store_true",
        help="Filter and collect only Ascend-related environment variables. Default: False."
    )
    extra_group.add_argument(
        "--user-config-path",
        metavar="",
        help="Path to the 'user_config.json' file for Kubernetes-based deployments."
    )
    extra_group.add_argument(
        "--mindie-env-path",
        metavar="",
        help="Path to the 'mindie_env.json' file for Kubernetes-based deployments."
    )
    extra_group.add_argument(
        "--mies-config-path",
        metavar="",
        help="Path to the 'config.json' file for daemon-based deployments."
    )
    extra_group.add_argument(
        "--rank-table-path",
        metavar="",
        help="Path to the rank table file. Supports both A2 and A3 formats."
    )
    extra_group.add_argument(
        "--weight-dir",
        metavar="",
        help="Directory path containing model weights."
    )
    extra_group.add_argument(
        "--chunk-size",
        metavar="",
        choices=[32, 64, 128, 256],
        type=int,
        default=32,
        help=(
            "Specify the chunk size (in KB) for calculating sha256sum of model tensors. "
            "Only tensors will be checksummed if this option is set. "
            "Supported values: 32, 64, 128, 256."
        )
    )
