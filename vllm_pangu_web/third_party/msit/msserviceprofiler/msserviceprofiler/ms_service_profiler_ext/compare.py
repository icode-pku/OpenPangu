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

import os
import re
import sqlite3
import argparse
import shutil
from pathlib import Path
from contextlib import contextmanager

import pandas as pd

from msserviceprofiler.msguard import validate_args, Rule
from msserviceprofiler.msguard.security import mkdir_s, open_s


@contextmanager
def connect_db(db_path):
    connection = None    
    try:
        connection = sqlite3.connect(db_path)
        yield connection
        connection.commit()
    except:
        if connection:
            connection.rollback()
        raise
    finally:
        if connection:
            connection.close()


def process_file_pairs(file_pairs, db_conn, excel_writer):
    from ms_service_profiler.utils.log import logger
    from msserviceprofiler.ms_service_profiler_ext.compare_tools import CSVComparator, DBComparator

    comparators = {'.csv': CSVComparator, '.db': DBComparator}

    for file_a, file_b in file_pairs:
        ext = Path(file_a).suffix.lower()
        logger.info("Begin to compare %r and %r", file_a, file_b)
        for comparator_cls in comparators.values():
            if not comparator_cls.supports(ext):
                continue
            comparator = comparator_cls(db_conn, excel_writer)
            try:
                comparator.process(file_a, file_b)
            except Exception as e:
                logger.error(
                    "During comparing %r and %r, there is an error ocurred: %r", file_a, file_b, e
                )
            logger.info("End to compare %r and %r", file_a, file_b)


def process_files(file_pairs, output_db, output_excel):
    with connect_db(output_db) as db_conn:
        with pd.ExcelWriter(output_excel, engine='openpyxl') as excel_writer:
            process_file_pairs(file_pairs, db_conn, excel_writer)

    shutil.copy(
        Path(__file__).parent / 'compare_tools' / 'compare_visualization.json',
        Path(output_db).with_name("compare_visualization.json")
    )


def arg_parse(subparsers):
    parser = subparsers.add_parser(
        "compare", formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="MS Server Profiler Compare Tool"
    )
    parser.add_argument(
        "input_path",
        type=validate_args(Rule.input_dir_traverse),
        help="Directory containing analyzed results"
    )
    parser.add_argument(
        "golden_path",
        type=validate_args(Rule.input_dir_traverse),
        help="Directory containing analyzed results"
    )
    parser.add_argument(
        "--output-path",
        type=validate_args(Rule.input_dir_traverse, fall_back_fn=mkdir_s),
        default=os.path.join(os.getcwd(), 'compare_result'),
        help="Output Directory after comparing."
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'fatal', 'critical'],
        help='Log level to print.'
    )
    parser.set_defaults(func=main)


def main(args):
    from ms_service_profiler.utils.log import set_log_level, logger
    from msserviceprofiler.ms_service_profiler_ext.compare_tools.collector import FileCollector

    set_log_level(args.log_level)

    result_prefix = os.path.join(args.output_path, 'compare_result')
    file_collector = FileCollector(
        pattern=re.compile(r'(batch|service|request)_summary\.csv|profiler\.db'),
        max_iter=100
    )

    file_pairs = file_collector.collect_pairs(args.input_path, args.golden_path)
    if not file_pairs:
        logger.warning("No files to compare, please check the input directories")
        return
    
    with open_s(f'{result_prefix}.db', 'w', encoding='utf-8'):
        with open_s(f'{result_prefix}.xlsx', 'w', encoding='utf-8'):
            process_files(file_pairs, f'{result_prefix}.db', f'{result_prefix}.xlsx')

    logger.info("Comparing finished successfully, the results stored under %r", args.output_path)
    logger.info("\nWhat's Next?\n\tYou may use the `grafana` to have a better visualization of the comparison results")
