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
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
from msserviceprofiler.ms_service_profiler_ext.compare import connect_db, main, arg_parse
from msserviceprofiler.ms_service_profiler_ext.compare_tools import CSVComparator
from msserviceprofiler.ms_service_profiler_ext.compare import connect_db, process_files
from msserviceprofiler.ms_service_profiler_ext.common.csv_fields import ServiceCSVFields


def build_test_data_df(value_list):
    test_data = {"Metric": ["m1", "m2"]}
    test_data[ServiceCSVFields.VALUE] = value_list
    return pd.DataFrame(test_data)


# Test cases for compare csv
def test_compare_csv_collectly(tmp_path):
    output_db = tmp_path / "output.db"
    output_excel = tmp_path / "output.xlsx"

    input_a = tmp_path / "input_a" / "service_summary.csv"
    input_b = tmp_path / "input_b" / "service_summary.csv"
    input_a.parent.mkdir(exist_ok=True)
    input_b.parent.mkdir(exist_ok=True)
    df_a = build_test_data_df([10, 20])
    df_b = build_test_data_df([15, 25])
    df_a.to_csv(input_a, index=False)
    df_b.to_csv(input_b, index=False)

    file_pairs = [[input_a, input_b]]
    with connect_db(output_db) as db_conn:
        with pd.ExcelWriter(output_excel, engine="openpyxl") as excel_writer:
            for file_a, file_b in file_pairs:
                comparator = CSVComparator(db_conn, excel_writer)
                comparator.process(file_a, file_b)


# Test cases for compare csv
def test_compare_csv_missing_file(tmp_path):
    output_db = tmp_path / "output.db"
    output_excel = tmp_path / "output.xlsx"

    file_a = tmp_path / "input_a" / "service_summary.csv"
    file_b = tmp_path / "input_b" / "service_summary.csv"
    file_a.parent.mkdir(exist_ok=True)
    file_b.parent.mkdir(exist_ok=True)
    df_a = build_test_data_df([10, 20])
    df_a.to_csv(file_a, index=False)

    with connect_db(output_db) as db_conn:
        with pd.ExcelWriter(output_excel, engine="openpyxl") as excel_writer:
            comparator = CSVComparator(db_conn, excel_writer)
            comparator.process(file_a, file_b)


# Test cases for compare csv
def test_compare_csv_wrong_value(tmp_path):
    output_db = tmp_path / "output.db"
    output_excel = tmp_path / "output.xlsx"

    file_a = tmp_path / "input_a" / "service_summary.csv"
    file_b = tmp_path / "input_b" / "service_summary.csv"
    file_a.parent.mkdir(exist_ok=True)
    file_b.parent.mkdir(exist_ok=True)
    df_a = build_test_data_df([10, 20])
    df_b = build_test_data_df([15, "25"])
    df_a.to_csv(file_a, index=False)
    df_b.to_csv(file_b, index=False)

    with connect_db(output_db) as db_conn:
        with pd.ExcelWriter(output_excel, engine="openpyxl") as excel_writer:
            comparator = CSVComparator(db_conn, excel_writer)
            comparator.process(file_a, file_b)


# Test cases for main
def test_main_given_valid_args_when_run_then_success(tmp_path):
    # Arrange
    input_a = tmp_path / "input_a"
    input_b = tmp_path / "input_b"
    Path(input_a).mkdir(exist_ok=True)
    Path(input_b).mkdir(exist_ok=True)
    df_a = build_test_data_df([10, 20])
    df_b = build_test_data_df([15, 25])
    csv_a = Path(input_a) / "service_summary.csv"
    csv_b = Path(input_b) / "service_summary.csv"
    df_a.to_csv(csv_a, index=False)
    df_b.to_csv(csv_b, index=False)
    input_a.chmod(0o750)
    input_b.chmod(0o750)
    csv_a.chmod(0o640)
    csv_b.chmod(0o640)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    arg_parse(subparsers)
    args = [str(input_a), str(input_b), "--output-path", str(tmp_path), "--log-level", "info"]

    # Act
    with patch("sys.argv", ["", "compare"] + args):
        # Add our advisor subparser
        main(parser.parse_args())

    # Assert
    assert (tmp_path / "compare_result.xlsx").exists()
    assert (tmp_path / "compare_result.db").exists()
