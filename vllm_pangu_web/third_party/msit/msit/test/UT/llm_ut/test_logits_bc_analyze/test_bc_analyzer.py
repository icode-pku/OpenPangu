import os
import warnings
from unittest import TestCase
import pandas as pd

from msit_llm.badcase_analyze.bad_case_analyze import BadCaseAnalyzer


class TestLogitsBadCaseAnalyzer(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = 'golden.csv'
        cls.golden_dict = {
            "key": [0, 1, 2, 3],
            "queries": ["How are you?", "Hello", "What is your name?", "What time is it?"],
            "input_token_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
            "output_token_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
            "passed": ["Correct", "Wrong", "Correct", "Correct"],
        }
        golden_df = pd.DataFrame(cls.golden_dict)
        golden_df.to_csv(cls.golden, encoding='utf-8', index=False)

        cls.test = 'test.csv'
        cls.test_dict = {
            "key": [0, 1, 2, 3],
            "queries": ["How are you?", "Hello", "What is your name?", "What time is it?"],
            "input_token_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
            "output_token_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
            "passed": ["Wrong", "Correct", "Wrong", "Correct"],
        }
        test_df = pd.DataFrame(cls.test_dict)
        test_df.to_csv(cls.test, encoding='utf-8', index=False)

        cls.my_path_with_wrong_cols = "my_path_with_wrong_cols.csv"
        cls.my_path_with_wrong_cols_dict = {
            "key": [0, 1, 2, 3],
            "queries": ["How are you?", "What is your name?", "What time is it?", "No!"],
            "input_token_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
            "output_token_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
        }
        my_path_with_wrong_cols_df = pd.DataFrame(cls.my_path_with_wrong_cols_dict)
        my_path_with_wrong_cols_df.to_csv(cls.my_path_with_wrong_cols, encoding='utf-8', index=False)

        cls.my_path_unmatched_content = "my_path_unmatched_df.csv"
        cls.my_path_unmatched_content_dict = {
            "key": [0, 1, 2, 3],
            "queries": ["How are you?", "What is your name?", "What time is it?", "No!"],
            "input_token_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
            "output_token_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
            "passed": ["Wrong", "Correct", "Wrong", "Correct"],
        }
        my_path_unmatched_content_df = pd.DataFrame(cls.my_path_unmatched_content_dict)
        my_path_unmatched_content_df.to_csv(cls.my_path_unmatched_content, encoding='utf-8', index=False)

        cls.analyzer = BadCaseAnalyzer()

    def test_logits_badcase_analyzer_for_right_csv_path(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
        self.analyzer.analyze(golden_csv_path=self.golden, test_csv_path=self.test)

        self.assertTrue(os.path.exists("msit_bad_case/analyzer"))
        self.assertTrue(
            any(
                file_name.endswith('.csv') 
                for file_name in os.listdir('msit_bad_case/analyzer')
            )
        )
    
    def test_logits_badcase_analyzer_for_csv_arg_not_str(self):
        with self.assertRaises(ValueError) as cm:
            self.analyzer.analyze(2, self.test)
            logger_output = str(cm.exception)
            self.assertIsNotNone(logger_output)
            self.assertRegex(logger_output, r'Invalid csv path, only path with suffix')

    def test_logits_badcase_analyzer_for_csv_arg_not_exist(self):
        with self.assertRaises(FileNotFoundError) as cm:
            self.analyzer.analyze(golden_csv_path='not_exist_path.csv', test_csv_path=self.test)
            logger_output = str(cm.exception)
            self.assertIsNotNone(logger_output)
            self.assertRegex(logger_output, r'not_exist_path.csv')

    def test_logits_badcase_analyzer_for_csv_with_wrong_cols(self):
        with self.assertRaises(KeyError) as cm:
            self.analyzer.analyze(self.golden, self.my_path_with_wrong_cols)
            logger_output = str(cm.exception)
            self.assertIsNotNone(logger_output)
            self.assertRegex(logger_output, r'Unmatched csv columns, expected to have')

    def test_logits_badcase_analyzer_unmatched_df(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
        with self.assertLogs('msit_logger', 'WARNING') as cm:
            self.analyzer.analyze(golden_csv_path=self.golden, test_csv_path=self.my_path_unmatched_content)
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r'below is a partial display of these unmatched')

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.golden)
        os.remove(cls.test)
        os.remove(cls.my_path_with_wrong_cols)
        os.remove(cls.my_path_unmatched_content)

        if os.path.exists("msit_bad_case"):
            import shutil

            shutil.rmtree("msit_bad_case")
