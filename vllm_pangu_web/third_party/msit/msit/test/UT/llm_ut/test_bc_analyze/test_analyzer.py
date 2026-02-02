import os
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from msit_llm import Analyzer, Synthesizer


class TestSynthezier(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = 'golden.csv'
        cls.golden_dict = {
            "queries": ["How are you?", "Hello", "What is your name?", "What time is it?"],
            "input_token_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
            "output_token_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
            "passed": ["Correct", "Wrong", "Correct", "Correct"],
        }
        golden_df = pd.DataFrame(cls.golden_dict)
        golden_df.to_csv(cls.golden, encoding='utf-8', index=False)

        cls.test = 'test.csv'
        cls.test_dict = {
            "queries": ["How are you?", "Hello", "What is your name?", "What time is it?"],
            "input_token_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
            "output_token_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
            "passed": ["Wrong", "Correct", "Wrong", "Correct"],
        }

        test_df = pd.DataFrame(cls.test_dict)
        test_df.to_csv(cls.test, encoding='utf-8', index=False)
        cls.analyzer = Analyzer()

    def test_analyzer_from_csv(self):
        self.analyzer.analyze(golden=self.golden, test=self.test)

        self.assertTrue(os.path.exists("msit_bad_case/analyzer"))
        self.assertTrue(
            any(
                file_name.endswith('.csv') 
                for file_name in os.listdir('msit_bad_case/analyzer')
            )
        )
    
    def test_analyzer_from_csv_arg_not_str(self):
        self.assertRaises(ValueError, self.analyzer.analyze, golden=2, test=self.test)

    def test_analyzer_from_csv_arg_not_exist(self):
        self.assertRaises(OSError, self.analyzer.analyze, golden='aqewwqe.csv', test=self.test)
    
    def test_analyzer_from_mixed(self):
        import time
        time.sleep(1) # The analyze is so fast that the time stamp does not change, leads to permssion error
        self.analyzer.analyze(golden=Synthesizer(**self.golden_dict), test=self.test)
        time.sleep(1)
        self.analyzer.analyze(golden=self.golden, test=Synthesizer(**self.test_dict))

        self.assertTrue(os.path.exists("msit_bad_case/analyzer"))
        self.assertTrue(
            any(
                file_name.endswith('.csv') 
                for file_name in os.listdir('msit_bad_case/analyzer')
            )
        )
        time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.golden)
        os.remove(cls.test)

        if os.path.exists("msit_bad_case"):
            import shutil

            shutil.rmtree("msit_bad_case")
