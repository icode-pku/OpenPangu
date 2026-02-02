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

import os

import pandas as pd

from components.utils.file_utils import (
    create_directory, check_path_readability, check_path_writability, write_df_to_csv
)
from msit_llm.bc_analyze.utils import get_timestamp
from msit_llm.common.constant import MSIT_BAD_CASE_FOLDER_NAME
from msit_llm.common.log import logger
from msit_llm.common.utils import load_file_to_read_common_check


class Analyzer(object):
    """This class is used for analyzing the bad case between `golden` and `test` results. `golden` or `test` result
    can be a csv path or a command. If it is a command, `Analyzer` will internally call `Synthesizer.from_cmd` to
    collect the result for you, and analyze it immediately.
    """
    ANALYZER_FOLDER_NAME = os.path.join(MSIT_BAD_CASE_FOLDER_NAME, 'analyzer')

    @staticmethod
    def analyze(golden, test) -> None:
        """This method is designed in case users collect the evaluation result in memory. Asides from path like,
        both `golden` and `test` can be an instance of `Synthesizer`.

        Parameter
        ---------
        `golden` : str or Synthesizer
            The csv path or Sythesizer instance that is considered to be the golden standard.
        `test` : str
            The csv path or Sythesizer instance that is considered to be the test result.

        All the notes, exceptions are consistent to `from_csv`

        Examples
        --------
        >>> from msit_llm import Synthezier, Analyzer
        >>> golden_synthesizer = Synthesizer(
        ...     queries='Question 1',
        ...     input_token_ids=[1, 2, 3],
        ...     output_token_ids=[4, 5, 6],
        ...     passed='Correct')
        >>> Analyzer.from_mixed(golden_synthesizer, test_csv_path)
        2024-08-07 04:45:13,123 - msit_llm_logger - INFO - Checking if the header of csv is valid...
        2024-08-07 04:45:14,546 - msit_llm_logger - INFO - Checking if path 'test_csv_path' is valid...
        2024-08-07 04:45:15,523 - msit_llm_logger - INFO - Checking if the header of csv is valid...
        2024-08-07 04:45:16,166 - msit_llm_logger - INFO - Analyzing...
        2024-08-07 04:45:13,651 - msit_llm_logger - INFO - 'Analyzer' has successfully finished the analysis,
        the result is stored at 'msit_bad_case_analyze/msit_bad_case_result_ieqwe2q5_20240720042235.csv'
        """
        from msit_llm.bc_analyze.synthezier import Synthesizer

        if isinstance(golden, Synthesizer):
            golden = golden.to_df(errors='trunc')
        else:
            golden = Analyzer._validate_csv_path(golden)

        if isinstance(test, Synthesizer):
            test = test.to_df(errors='trunc')
        else:
            test = Analyzer._validate_csv_path(test)

        Analyzer._validate_df(golden)
        Analyzer._validate_df(test)

        Analyzer._compare_golden_with_test(golden, test)

    @staticmethod
    def _validate_csv_path(csv_path: str) -> pd.DataFrame:
        logger.info("Checking if path %r is valid...", csv_path)

        if not isinstance(csv_path, str) or not csv_path.endswith('.csv'):
            logger.error("Invalid csv path, only path with suffix '.csv' is allowed: %r", csv_path)
            raise ValueError

        csv_path = load_file_to_read_common_check(csv_path)

        return pd.read_csv(csv_path, encoding='utf-8')

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        logger.info("Checking if the header of csv is valid...")
        df.rename(columns={'pass': 'passed'}, inplace=True)
        df.dropna(inplace=True)

        main_columns = ['queries', 'input_token_ids', 'output_token_ids', 'passed']
        if any(column not in df.columns for column in main_columns):
            logger.error(
                "Unmatched csv columns, expected to have 'queries', 'input_token_ids', 'output_token_ids', and 'passed'"
            )
            raise KeyError

    @staticmethod
    def _compare_golden_with_test(golden_df, test_df) -> None:
        logger.info("Analyzing...")
        merged_df = golden_df.merge(test_df, on='queries', how='left', suffixes=('_golden', '_test'), indicator=True)

        not_equal_mask = merged_df['passed_golden'] != merged_df['passed_test']
        bad_case_mask = not_equal_mask & (merged_df['_merge'] == 'both')
        filtered_df = merged_df[bad_case_mask]

        desired_columns = ['queries']
        for column in ['input_token_ids', 'output_token_ids', 'passed']:
            desired_columns.append(f'{column}_golden')
            desired_columns.append(f'{column}_test')

        Analyzer._save_result(filtered_df[desired_columns])

        not_both_mask = merged_df['_merge'] != 'both'
        not_both_count = not_both_mask.sum()
        if not_both_count != 0:
            unmatched_queries = merged_df['queries'][not_both_mask].head()
            logger.warning(
                "There are '%s' quer(ies) not matched, below is a partial display of these unmatched queries:\n\t '%s'",
                not_both_count,
                unmatched_queries.to_string(header=False)
            )

    @staticmethod
    def _save_result(df_to_save: pd.DataFrame) -> None:
        if df_to_save.empty:
            logger.warning(
                "'Analyzer' detected that there is no difference between the given golden result and test result. "
                "Hence no result is saved")
            return

        create_directory(Analyzer.ANALYZER_FOLDER_NAME)
        check_path_readability(Analyzer.ANALYZER_FOLDER_NAME)
        check_path_writability(Analyzer.ANALYZER_FOLDER_NAME)
        path = os.path.join(Analyzer.ANALYZER_FOLDER_NAME, f"{get_timestamp()}.csv")
        write_df_to_csv(df_to_save, path)

        logger.info("'Analyzer' has successfully finished the analysis, the result is stored at %r", path)
