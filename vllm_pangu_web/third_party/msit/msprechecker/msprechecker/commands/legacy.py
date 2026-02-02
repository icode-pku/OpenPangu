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

from ..utils import global_logger


def add_legacy_argument(parser: argparse.ArgumentParser, dump_mode=False):
    legacy_group = parser.add_argument_group(
        "Legacy Arguments (Deprecated)",
        "These arguments are maintained for backward compatibility and may be removed in future versions."
    )

    _add_legacy_args(legacy_group)
    if dump_mode:
        legacy_group.add_argument(
            "-d", "--dump_file_path",
            help="Path for saving envs (legacy parameter)"
        )


def show_legacy_warnings(args: argparse.Namespace):
    _warn_log_level(args)
    _warn_checkers(args)
    _warn_service_config_path(args)
    _warn_mindie_env_config_path(args)
    _warn_ranktable_file(args)
    _warn_sha256_blocknum(args)
    _warn_additional_checks_yaml(args)
    _warn_save_env(args)
    _warn_dump_file_path(args)


def _add_legacy_args(legacy_group):
    legacy_group.add_argument(
        "--log_level",
        choices=["info", "warning", "error", "debug"],
        help="Specify log level (legacy parameter)"
    )
    legacy_group.add_argument(
        "-ch", "--checkers",
        nargs="+",
        choices=["basic", "hccl", "model", "hardware", "all"],
        help="Specify checker types (legacy parameter)"
    )
    legacy_group.add_argument(
        "-service", "--service_config_path",
        help="MINDIE service or config json path (legacy parameter)"
    )
    legacy_group.add_argument(
        "-user", "--user_config_path",
        help="k8s user config json path (legacy parameter)"
    )
    legacy_group.add_argument(
        "--mindie_env_config_path",
        help="k8s mindie env config json path (legacy parameter)"
    )
    legacy_group.add_argument(
        "-ranktable", "--ranktable_file",
        help="HCCL rank table file path (legacy parameter)"
    )
    legacy_group.add_argument(
        "--weight_dir",
        help="Specify the model weight directory for model checks (legacy parameter)"
    )
    legacy_group.add_argument(
        "-blocknum", "--sha256_blocknum",
        type=int,
        help="Sampling file block number for checking sha256sum (legacy parameter)"
    )
    legacy_group.add_argument(
        "-add", "--additional_checks_yaml",
        help="additional checks replacing default checking values (legacy parameter)"
    )
    legacy_group.add_argument(
        "-s", "--save_env",
        help="Save env changes as a file (legacy parameter)"
    )


def _warn_log_level(args):
    if getattr(args, "log_level", None):
        global_logger.warning(
            "DeprecationWarning: The '--log_level' argument is deprecated. "
            "Please use '--severity-level' instead."
        )


def _warn_checkers(args):
    if getattr(args, "checkers", None):
        global_logger.warning(
            "DeprecationWarning: The '--checkers' argument is deprecated. "
            "HCCL will be checked whenever '--rank-table-path' is given, "
            "model is triggered by '--weight-dir', and hardware is triggered by '--hardware'. "
            "This API is maintained for backward compatibility."
        )
        args.hardware = "hardware" in args.checkers
        if "model" in args.checkers and not getattr(args, "weight_dir", None):
            global_logger.warning(
                "UsageWarning: Set '-ch model' or '-ch all' without providing '--weight-dir' " \
                "will not take affect"
            )


def _warn_service_config_path(args):
    if getattr(args, "service_config_path", None):
        global_logger.warning(
            "DeprecationWarning: The '--service_config_path' argument is deprecated. "
            "Please use '--mies-config-path' instead."
        )
        args.mies_config_path = args.service_config_path


def _warn_mindie_env_config_path(args):
    if getattr(args, "mindie_env_config_path", None):
        global_logger.warning(
            "DeprecationWarning: The '--mindie_env_config_path' argument is deprecated. "
            "Please use '--mindie-env-path' instead."
        )
        args.mindie_env_path = args.mindie_env_config_path


def _warn_ranktable_file(args):
    if getattr(args, "ranktable_file", None):
        global_logger.warning(
            "DeprecationWarning: The '--ranktable_file' argument is deprecated. "
            "Please use '--rank-table-path' instead."
        )
        args.ranktable_file = args.rank_table_path


def _warn_sha256_blocknum(args):
    if getattr(args, "sha256_blocknum", None):
        global_logger.warning(
            "DeprecationWarning: The '--sha256_blocknum' argument is deprecated. "
            "It is now migrated to 'msprechecker dump' option and use '--chunk-size' instead. "
            "No check will be applied. This API is maintained for backward compatibility."
        )
        args.chunk_size = args.sha256_blocknum


def _warn_additional_checks_yaml(args):
    if getattr(args, "additional_checks_yaml", None):
        global_logger.warning(
            "DeprecationWarning: The '--additional_checks_yaml' argument is deprecated. "
            "Please use '--custom-config-path' instead."
        )
        args.custom_config_path = args.additional_checks_yaml


def _warn_save_env(args):
    if getattr(args, "save_env", None):
        global_logger.warning(
            "DeprecationWarning: The '--save_env' argument is deprecated. "
            "Environment changes file will be stored under current working directory."
        )


def _warn_dump_file_path(args):
    if getattr(args, "dump_file_path", None):
        global_logger.warning(
            "DeprecationWarning: The '--dump_file_path' argument is deprecated. "
            "Please use '--output-path' instead."
        )
        args.output_path = args.dump_file_path
