# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from functools import partial

from ait_tensor_view.operation import SliceOperation, PermuteOperation
from ait_tensor_view.handler import handle_tensor_view
from components.utils.parser import BaseCommand
from components.utils.file_open_check import FileStat
from components.utils.util import load_file_to_read_common_check_for_cli


def check_output_path_legality(value):
    if not value:
        return value
    try:
        file_stat = FileStat(value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"output path:{value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal("write", strict_permission=False):
        raise argparse.ArgumentTypeError(f"output path:{value} can not write. Please check.")
    return value


def parse_operations(value):
    operations = value.split(";")

    ops = []

    for op in operations:
        if op.startswith("[") and op.endswith("]"):
            ops.append(SliceOperation(op))
        elif op.startswith("(") and op.endswith(")"):
            ops.append(PermuteOperation(op))
        else:
            raise SyntaxError(f"Invalid operation string: {op}")

    return ops


def get_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


class TensorViewCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "--bin", "-b",
            type=partial(
                    load_file_to_read_common_check_for_cli,
                    exts=['.bin', '.pth']
                 ),
            required=True,
            help="Bin file path"
        )

        parser.add_argument(
            "--print", "-p",
            action="store_true",
            help="print tensor"
        )

        parser.add_argument(
            "--output", "-o",
            type=check_output_path_legality,
            help="where the tensor should be saved (Default None means current directory)"
        )

        parser.add_argument(
            "--operations", "-op",
            type=parse_operations,
            help="Each operation is separated by a semicolon; slice operations should use square brackets; permute "
                 "sequences should use parentheses"
        )

    def handle(self, args):
        handle_tensor_view(args)


def get_cmd_instance():
    help_info = "view / slice / permute / save the dumped tensor"
    cmd_instance = TensorViewCommand("tensor-view", help_info)
    return cmd_instance
