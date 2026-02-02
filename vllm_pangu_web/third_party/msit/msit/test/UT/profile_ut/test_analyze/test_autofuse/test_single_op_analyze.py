import os
import unittest
import shutil
import subprocess
from unittest.mock import patch, MagicMock
from pandas.testing import assert_frame_equal
import numpy as np
import pandas as pd

from msit_prof.analyze.autofuse.single_op_analyze import SingleOpAnalyzer


class TestSingleOpAnalyzer(unittest.TestCase):
    INPUT_PATH = os.path.join(os.path.dirname(__file__), "test_single_op_analyze")
    OPS_MAPPING_JSON = [
            {
                'op': [
                    {
                        'type': 'AscBackend',
                        'name': 'fused_graph_50',
                        'attr': [
                            {'key': '_datadump_original_op_names', 'value':
                                {'list': {'s': ['Mul_7', 'Mul_6', 'Reshape_18', 'Reshape_19']}}},
                            {'key': '_datadump_original_op_types', 'value':
                                {'list': {'s': ['Mul', 'Mul', 'Reshape', 'Reshape']}}}
                        ]
                    },
                    {
                        'type': 'AscBackend',
                        'name': 'fused_graph_52',
                        'attr': [
                            {'key': '_datadump_original_op_names', 'value':
                                {'list': {'s': ['Reshape_23', 'Sigmoid', 'Reshape_22', 'mul_8', 'Sigmoid_1']}}},
                            {'key': '_datadump_original_op_types', 'value':
                                {'list': {'s': ['Reshape', 'Sigmoid', 'Reshape', 'Mul', 'Sigmoid']}}}
                        ]
                    }
                ]
            }
        ]

    def setUp(self):
        os.makedirs(self.INPUT_PATH, mode=0o750, exist_ok=True)
        self.args = MagicMock()
        self.args.fused = os.path.join(self.INPUT_PATH, "fused.csv")
        self.args.origin = os.path.join(self.INPUT_PATH, "origin.csv")
        self.args.output = self.INPUT_PATH
        self.args.ops_graph = [os.path.join(self.INPUT_PATH, "ge_proto_00000001_graph_31_Build.txt")]
        self.analyzer = SingleOpAnalyzer(self.args)
        fused_df = pd.DataFrame({
            'Op Name': ['fused_graph_50', 'fused_graph_52'] * 2,
            'OP Type': ['AscBackend', 'AscBackend'] * 2,
            'Task Duration(us)': [36.041, 4.1, 35.121, 3.42],
            'Input Shapes': ['4096,8;4096,8,256;4096,8', '4096,1;4096,1'] * 2,
            'Input Data Types': ['FLOAT;FLOAT;FLOAT', 'FLOAT;FLOAT'] * 2,
            'Output Shapes': ['4096,8,256;4096,8,256', '4096;4096;4096'] * 2,
            'Output Data Types': ['FLOAT;FLOAT', 'FLOAT;FLOAT;FLOAT'] * 2,
        })
        origin_df = pd.DataFrame({
            'Op Name': ['Mul_6', 'Mul_7', 'Sigmoid_1'] * 2,
            'OP Type': ['Mul', 'Mul', 'Sigmoid'] * 2,
            'Task Duration(us)': [21.56, 14.96, 1.82, 21.24, 14.94, 1.58],
            'Input Shapes': ["4096,8,1;4096,8,256", "4096,8,1;4096,8,256", "4096"] * 2,
            'Input Data Types': ['FLOAT;FLOAT', 'FLOAT;FLOAT', 'FLOAT'] * 2,
            'Output Shapes': ["4096,8,256", "4096,8,256", "4096"] * 2,
            'Output Data Types': ['FLOAT', 'FLOAT', 'FLOAT'] * 2,
        })
        fused_df.to_csv(self.args.fused, index=False)
        origin_df.to_csv(self.args.origin, index=False)

    def tearDown(self):
        if os.path.exists(self.INPUT_PATH):
            shutil.rmtree(self.INPUT_PATH)

    def test_calculate_gm_valid_inputs(self):
        test_cases = [
            (("INT32", "2,3"), sum([24 / 1024])),  # 24 / 1024表示HBM的计算方式，通过shape和type所占字节数计算数据量，单位用KB
            (("FLOAT;INT32", '"1,2,3";4'), sum([12 / 1024, 16 / 1024])),
            (("INT32;FLOAT", "5,5;10"), sum([100 / 1024, 20 / 1024])),
            (("FLOAT", "  100 , 200 "), sum([40000 / 1024])),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val, expected=expected):
                result = self.analyzer.calculate_gm(input_val[0], input_val[1])
                self.assertEqual(result, expected)

    def test_calculate_gm_empty_values(self):
        test_cases = [
            (("", "1,2")),
            (("INT32", "")), 
            (("", "")),
        ]
        for case in test_cases:
            with self.subTest(input=case):
                result = self.analyzer.calculate_gm(case[0], case[1])
                self.assertEqual(result, 0)

    @patch('msit_prof.analyze.autofuse.single_op_analyze.get_all_subgraph')
    @patch('subprocess.run')
    def test_convert_ge_graph_given_valid_command_when_running_atc_then_conversion_success(self, mock_run,
                                                                                           mock_get_all_subgraph):
        mock_run.return_value.returncode = 0
        mock_get_all_subgraph.return_value = self.OPS_MAPPING_JSON
        self.analyzer.convert_ge_graph()
        mock_run.assert_called_once()
        self.assertTrue(mock_run.call_args[1]['stdout'] is subprocess.PIPE)
        expected = pd.DataFrame({
            'fused_op_name': ['fused_graph_50'] * 4 + ['fused_graph_52'] * 5,
            'origin_op_name': ['Mul_7', 'Mul_6', 'Reshape_18', 'Reshape_19',
                               'Reshape_23', 'Sigmoid', 'Reshape_22', 'mul_8', 'Sigmoid_1'],
            'origin_op_type': ['Mul', 'Mul', 'Reshape', 'Reshape', 'Reshape', 'Sigmoid', 'Reshape', 'Mul', 'Sigmoid']
        })
        result = self.analyzer.fused_graph_to_origin_op_mapping
        self.assertTrue(result.equals(expected))

    @patch('subprocess.run')
    def test_convert_ge_graph_given_invalid_command_when_running_atc_then_conversion_fails(self, mock_run):
        mock_run.return_value.returncode = 1
        with self.assertRaises(RuntimeError):
            self.analyzer.convert_ge_graph()

    def test_load_op_summary_given_valid_csv_files_when_loading_then_dataframes_created(self):
        self.analyzer.load_op_summary()
        expected_fused_df = pd.DataFrame({
            'fused_op_name': ['fused_graph_50', 'fused_graph_52'] * 2,
            'fused_op_type': ['AscBackend', 'AscBackend'] * 2,
            'fused_duration': [36.041, 4.1, 35.121, 3.42],
            'fused_input_shapes': ['4096,8;4096,8,256;4096,8', '4096,1;4096,1'] * 2,
            'fused_input_data_types': ['FLOAT;FLOAT;FLOAT', 'FLOAT;FLOAT'] * 2,
            'fused_output_shapes': ['4096,8,256;4096,8,256', '4096;4096;4096'] * 2,
            'fused_output_data_types': ['FLOAT;FLOAT', 'FLOAT;FLOAT;FLOAT'] * 2,
        })
        expected_origin_df_columns = ['origin_op_name', 'OP Type', 'Task Duration(us)', 'Input Shapes',
                                      'Input Data Types', 'Output Shapes', 'Output Data Types']
        self.assertTrue(self.analyzer.fused_df.equals(expected_fused_df))
        self.assertEqual(len(self.analyzer.origin_df), 6)
        self.assertEqual(self.analyzer.origin_df.columns.tolist(), expected_origin_df_columns)

    @patch('msit_prof.analyze.autofuse.single_op_analyze.get_all_subgraph')
    @patch('subprocess.run')
    def test_build_fusion_origin_analysis_should_return_analysis_df_when_all_dfs_valid(self, mock_run,
                                                                                       mock_get_all_subgraph):
        mock_run.return_value.returncode = 0
        mock_get_all_subgraph.return_value = self.OPS_MAPPING_JSON
        self.analyzer.convert_ge_graph()
        self.analyzer.load_op_summary()
        result = self.analyzer.build_fusion_origin_analysis()
        self.assertEqual(result.columns.tolist(), ['fused_op_name', 'fused_op_type', 'fused_duration',
            'fused_input_shapes', 'fused_input_data_types', 'fused_output_shapes', 'fused_output_data_types',
            'origin_op_name', 'origin_op_type', 'OP Type', 'Task Duration(us)', 'Input Shapes', 'Input Data Types',
            'Output Shapes', 'Output Data Types', '_merge'])
        self.assertEqual(result.shape, (12, 16))

    @patch('msit_prof.analyze.autofuse.single_op_analyze.get_all_subgraph')
    @patch('subprocess.run')
    def test_save_analyze_result_given_valid_result_when_saving_then_csv_created(self, mock_run,
                                                                                 mock_get_all_subgraph):
        mock_run.return_value.returncode = 0
        mock_get_all_subgraph.return_value = self.OPS_MAPPING_JSON
        self.analyzer.analyze()
        profile_analysis_df = pd.read_csv(os.path.join(self.analyzer.output_path, "profile_analysis.csv"))
        expected_df = pd.DataFrame({
            'Fuse OpName': ['fused_graph_50', 'fused_graph_52'],
            'Fuse OpType': ['AscBackend', 'AscBackend'],
            'Origin Ops': ['(Mul_7, Mul); (Mul_6, Mul); (Reshape_18, Reshape); (Reshape_19, Reshape)',
                '(Reshape_23, Reshape); (Sigmoid, Sigmoid); (Reshape_22, Reshape); (mul_8, Mul); (Sigmoid_1, Sigmoid)'],
            'Fused Durations(us)': [71.162, 7.52],
            'Origin Durations(us)': [72.7, 3.4],
            'Time Ratio': [0.97884, np.nan],
            'Time Difference': [-1.538, np.nan],
            'HBMs Difference': ['(input:-16384.0, output:0.0)', np.nan],
            'HBMs Ratio': ['(input:0.5019455252918288, output:1.0)', np.nan],
            'Fused HBMs(KB)': ['(input:16512.0, output:32768.0)', '(input:16.0, output:24.0)'],
            'Origin Duration(us) Each Op': ['(Mul_6, 42.8); (Mul_7, 29.9); (Reshape_18, 0.0); (Reshape_19, 0.0)',
                                            '(Reshape_22, 0.0); (Reshape_23, 0.0); (Sigmoid, 0.0); '
                                            '(Sigmoid_1, 3.4000000000000004); (mul_8, 0.0)'],
            'Origin HBMs Each Op(KB)': ['(Mul_6, input:16448.0, output:16384.0); '
                                        '(Mul_7, input:16448.0, output:16384.0); '
                                        '(Reshape_18, input:0.0, output:0.0); '
                                        '(Reshape_19, input:0.0, output:0.0)',
                                        '(Reshape_22, input:0.0, output:0.0); '
                                        '(Reshape_23, input:0.0, output:0.0); '
                                        '(Sigmoid, input:0.0, output:0.0); '
                                        '(Sigmoid_1, input:8.0, output:8.0); '
                                        '(mul_8, input:0.0, output:0.0)'],
            'Origin HBMs Total(KB)': ['(input:32896.0, output:32768.0)', '(input:8.0, output:8.0)'],
            'Not Found Origin Op': [np.nan, 'Sigmoid; mul_8']
        })
        assert_frame_equal(profile_analysis_df, expected_df)
