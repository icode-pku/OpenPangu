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
from pathlib import Path


def main():
    from msserviceprofiler.ms_service_profiler_ext import compare, split, analyze
    from msserviceprofiler.msservice_advisor import advisor
    from msserviceprofiler.modelevalstate.train import source_to_train
    from msserviceprofiler.modelevalstate.optimizer import optimizer
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="[MindStudio] msserviceprofiler command line tool"
    )
    subparsers = parser.add_subparsers(help="sub-command help")

    source_to_train.arg_parse(subparsers)
    optimizer.arg_parse(subparsers)

    advisor.arg_parse(subparsers)

    analyze.arg_parse(subparsers)
    split.arg_parse(subparsers)
    compare.arg_parse(subparsers)
    args = parser.parse_args()

    # run
    if hasattr(args, "func"):
        args.func(args=args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()