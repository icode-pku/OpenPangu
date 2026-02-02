# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd. All rights reserved.
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
from datetime import datetime, timedelta, timezone

import pandas as pd

from msit_llm.common.log import logger
from msit_llm.common.constant import MSIT_BAD_CASE_FOLDER_NAME
from msit_llm.common.utils import load_file_to_read_common_check
from components.utils.file_open_check import ms_open, sanitize_cell_for_dataframe
from components.utils.security_check import ms_makedirs
from components.utils.constants import CSV_FILE_MAX_SIZE


class BadCaseAnalyzer(object):
    """This class is used for analyzing the bad case between `golden` and `test` results. `golden` or `test` result
    must be a csv path.
    """
    ANALYZER_FOLDER_NAME = os.path.join(MSIT_BAD_CASE_FOLDER_NAME, 'analyzer')
    BAD_CASE_CSV_PREFIX = 'msit_bad_case_result_'

    @classmethod
    def analyze(cls, golden_csv_path, test_csv_path):
        """Analyze the bad case by comparing `golden_csv_path` with `test_csv_path`, the analysis will be stored 
        as a csv file. If there is no difference between these two, then no report will be saved.
        
        Paramters
        ---------
        `golden_csv_path` : str
            The csv path that is considered to be the golden standard.
        `test_csv_path` : str
            The csv path that is considered to be the test result.

        """
        logger.info(
            "'Analyzer' received two csv paths, the golden one is:\n\t%r\nand the test one is:\n\t%r" % 
            (golden_csv_path, test_csv_path)
        )

        golden_df = cls._validate_csv_path(golden_csv_path)
        test_df = cls._validate_csv_path(test_csv_path)

        cls._validate_df(golden_df)
        cls._validate_df(test_df)

        cls._compare_golden_with_test(golden_df, test_df)
    
    @classmethod
    def _validate_csv_path(cls, csv_path: str):
        logger.info("Checking if path %r is valid..." % csv_path)
        if not isinstance(csv_path, str) or not csv_path.endswith('.csv'):
            raise ValueError("Invalid csv path, only path with suffix '.csv' is allowed: %r" % csv_path)
            
        csv_path = load_file_to_read_common_check(csv_path)
        return pd.read_csv(csv_path, encoding='utf-8')

    @classmethod
    def _validate_df(cls, df: pd.DataFrame):
        logger.info("Checking if the header of csv is valid...")
        df.rename(columns={'pass': 'passed'}, inplace=True)
        df.dropna(inplace=True)
        main_columns = ['key', 'passed']
        
        df['passed'] = df['passed'].astype(str).str.lower().replace({'correct': 'true', 'wrong': 'false'})
        if any(column not in df.columns for column in main_columns):
            raise KeyError(
                "Unmatched csv columns, expected to have 'key' and 'passed'"
            )
    
    @classmethod
    def _compare_golden_with_test(cls, golden_df, test_df):
        logger.info("Analyzing...")
        all_columns = golden_df.columns.tolist()
        if 'queries' in all_columns:
            merged_col = ['key', 'queries']
        else:
            merged_col = ['key']
        if 'input_token_ids' in all_columns and 'output_token_ids' in all_columns:
            others_column = ['input_token_ids', 'output_token_ids', 'passed']
        else:
            others_column = ['passed']
        merged_df = golden_df.merge(test_df, on=merged_col, how='left', suffixes=('_golden', '_test'), indicator=True)

        not_equal_mask = (merged_df['passed_golden'] != merged_df['passed_test'])
        bad_case_mask = not_equal_mask & (merged_df['_merge'] == 'both')
        filtered_df = merged_df[bad_case_mask]

        for column in others_column:
            merged_col.append(f'{column}_golden')
            merged_col.append(f'{column}_test')

        cls._save_result(filtered_df[merged_col])

        not_both_mask = merged_df['_merge'] != 'both'
        not_both_count = not_both_mask.sum()
        if not_both_count != 0:
            unmatched_queries = merged_df['key'][not_both_mask].head()
            logger.warning(
                "There are '%s' quer(ies) not matched, "
                "below is a partial display of these unmatched query's key:\n\t '%s'", 
                not_both_count,
                unmatched_queries.to_string(header=False)
            )
        
    @classmethod
    def _save_result(cls, df_to_save: pd.DataFrame, suffix='.csv'):
        if df_to_save.empty:
            logger.warning(
                "'Analyzer' detected that there is no difference between the given golden result and test result. "
                "Hence no result is saved")
            return
        
        sanitize_cell_for_dataframe(df_to_save)
        
        ms_makedirs(cls.ANALYZER_FOLDER_NAME, mode=0o700, exist_ok=True)
        path = os.path.join(cls.ANALYZER_FOLDER_NAME, cls._get_candidate_path(suffix=suffix))

        with ms_open(path, 'w', max_size=CSV_FILE_MAX_SIZE) as file:
            df_to_save.to_csv(file, encoding='utf-8', index=False)

        logger.info("'Analyzer' has successfully finished the analysis, the result is stored at %r" % path)

    @classmethod
    def _get_candidate_path(cls, suffix):
        return cls.BAD_CASE_CSV_PREFIX + get_timestamp() + suffix


def get_timestamp():
    cst_timezone = timezone(timedelta(hours=8))
    current_time = datetime.now(cst_timezone)
    return current_time.strftime("%Y%m%d%H%M%S")
