import argparse
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

from components.utils.file_utils import FileCheckException


# Patch import for all command dependencies
MODULE_PATH = "msit_llm.__main__"
FILE_PATH = os.path.realpath(__file__)
PARENT_DIR = os.path.dirname(FILE_PATH)


class TestDumpCommand(unittest.TestCase):
    @patch(f"{MODULE_PATH}.set_log_level")
    @patch(f"{MODULE_PATH}.logger")
    @patch(f"{MODULE_PATH}.init_dump_task")
    @patch(f"{MODULE_PATH}.clear_dump_task")
    @patch(f"{MODULE_PATH}.is_enough_disk_space_left", return_value=True)
    @patch(f"{MODULE_PATH}.filter_cmd", side_effect=lambda x: x)
    @patch(f"{MODULE_PATH}.subprocess.run")
    def test_handle_exec(self, mock_run, mock_filter_cmd, mock_disk, mock_clear, mock_init, mock_logger, mock_set_log):
        from msit_llm.__main__ import DumpCommand
        args = argparse.Namespace(
            exec="echo hello",
            log_level="info",
            output=".",
            save_desc=False, ids="", range="0,0", child=True, time=3, opname=None,
            tiling=False, save_tensor_part=2, type=['tensor', 'model'], device_id=None,
            set_random_seed=None, enable_symlink=False, config_path=""
        )
        DumpCommand("dump", "help").handle(args)
        mock_set_log.assert_called_once_with("info")
        mock_init.assert_called_once()
        mock_run.assert_called_once()
        mock_clear.assert_called_once()

    @patch(f"{MODULE_PATH}.is_enough_disk_space_left", return_value=False)
    @patch(f"{MODULE_PATH}.set_log_level")
    @patch(f"{MODULE_PATH}.logger")
    @patch(f"{MODULE_PATH}.init_dump_task")
    def test_handle_exec_disk_full(self, mock_init, mock_logger, mock_set_log, mock_disk):
        from msit_llm.__main__ import DumpCommand
        args = argparse.Namespace(
            exec="echo hello",
            log_level="info",
            output=".",
            save_desc=False, ids="", range="0,0", child=True, time=3, opname=None,
            tiling=False, save_tensor_part=2, type=['tensor', 'model'], device_id=None,
            set_random_seed=None, enable_symlink=False
        )
        with self.assertRaises(OSError):
            DumpCommand("dump", "help").handle(args)

    @patch(f"{MODULE_PATH}.set_log_level")
    @patch("components.utils.cmp_algorithm.register_custom_compare_algorithm")
    @patch("msit_llm.compare.torchair_acc_cmp.get_torchair_ge_graph_path", return_value="some_path")
    @patch("msit_llm.compare.torchair_acc_cmp.acc_compare")
    def test_handle_torchair_ge_graph(self, mock_acc_compare, mock_get_graph, mock_register, mock_set_log):
        from msit_llm.__main__ import CompareCommand
        args = argparse.Namespace(
            golden_path=FILE_PATH, my_path=FILE_PATH, cmp_level=None, output=".", mapping_file="",
            custom_algorithms=None, log_level="info", weight=False, stats=False, rank_id=None
        )
        CompareCommand("compare", "help").handle(args)
        mock_acc_compare.assert_called_once()

    @patch(f"{MODULE_PATH}.set_log_level")
    @patch("components.utils.cmp_algorithm.register_custom_compare_algorithm")
    @patch("msit_llm.compare.torchair_acc_cmp.get_torchair_ge_graph_path", return_value=None)
    @patch("msit_llm.compare.cmp_mgr.CompareMgr")
    def test_handle_compare_mgr(self, mock_mgr, mock_get_graph, mock_register, mock_set_log):
        from msit_llm.__main__ import CompareCommand
        mock_mgr_instance = MagicMock()
        mock_mgr.return_value = mock_mgr_instance
        mock_mgr_instance.is_parsed_cmp_path.return_value = True
        args = argparse.Namespace(
            golden_path=PARENT_DIR, my_path=PARENT_DIR, cmp_level=None, output=".", mapping_file="",
            custom_algorithms=None, log_level="info", weight=False, stats=False
        )
        CompareCommand("compare", "help").handle(args)
        mock_mgr_instance.compare.assert_called_once()

    @patch(f"{MODULE_PATH}.set_log_level")
    @patch("components.utils.cmp_algorithm.register_custom_compare_algorithm")
    @patch("msit_llm.compare.torchair_acc_cmp.get_torchair_ge_graph_path", return_value=None)
    @patch("msit_llm.compare.cmp_weight.compare_weight")
    def test_handle_weight(self, mock_cmp_weight, mock_get_graph, mock_register, mock_set_log):
        from msit_llm.__main__ import CompareCommand
        args = argparse.Namespace(
            golden_path=PARENT_DIR, my_path=PARENT_DIR, cmp_level=None, output=".", mapping_file="",
            custom_algorithms=None, log_level="info", weight=True, stats=False
        )
        CompareCommand("compare", "help").handle(args)
        mock_cmp_weight.assert_called_once()


class TestErrCheck(unittest.TestCase):
    @patch(f"{MODULE_PATH}.set_log_level")
    @patch(f"{MODULE_PATH}.process_error_check")
    def test_handle(self, mock_process, mock_set_log):
        from msit_llm.__main__ import ErrCheck
        args = argparse.Namespace(exec="run", type=["overflow"], output="", exit=False, log_level="info")
        ErrCheck("errcheck", "help").handle(args)
        mock_process.assert_called_once()


class TestBCAnalyze(unittest.TestCase):
    @patch(f"{MODULE_PATH}.set_log_level")
    @patch(f"{MODULE_PATH}.Analyzer.analyze")
    def test_handle(self, mock_analyze, mock_set_log):
        from msit_llm.__main__ import BCAnalyze
        args = argparse.Namespace(golden="golden.csv", test="test.csv", log_level="info")
        with self.assertRaises(FileCheckException) as context:
            BCAnalyze("analyze", "help").handle(args)
        self.assertEqual(str(context.exception),
                         FileCheckException.err_strs.get(FileCheckException.ILLEGAL_PATH_ERROR))
        mock_analyze.assert_not_called()


class TestBadCaseAnalyze(unittest.TestCase):
    @patch(f"{MODULE_PATH}.set_log_level")
    @patch(f"{MODULE_PATH}.BadCaseAnalyzer.analyze")
    def test_handle(self, mock_analyze, mock_set_log):
        from msit_llm.__main__ import BadCaseAnalyze
        args = argparse.Namespace(golden_path="golden.csv", my_path="my.csv", log_level="info")
        with self.assertRaises(FileCheckException) as context:
            BadCaseAnalyze("bcanalyze", "help").handle(args)
        self.assertEqual(str(context.exception),
                         FileCheckException.err_strs.get(FileCheckException.ILLEGAL_PATH_ERROR))
        mock_analyze.assert_not_called()


class TestLogitsDump(unittest.TestCase):
    @patch(f"{MODULE_PATH}.set_log_level")
    @patch("msit_llm.logits_dump.logits_dump.LogitsDumper")
    def test_handle(self, mock_dumper, mock_set_log):
        from msit_llm.__main__ import LogitsDump
        args = argparse.Namespace(exec="run", bad_case_result_csv="bad.csv", token_range=1, log_level="info")
        with self.assertRaises(FileCheckException) as context:
            LogitsDump("logitsdump", "help").handle(args)
        self.assertEqual(str(context.exception),
                         FileCheckException.err_strs.get(FileCheckException.ILLEGAL_PATH_ERROR))
        mock_dumper.return_value.dump_logits.assert_not_called()


class TestLogitsCompare(unittest.TestCase):
    @patch(f"{MODULE_PATH}.set_log_level")
    @patch("msit_llm.logits_compare.logits_cmp.LogitsComparison")
    def test_handle(self, mock_cmp, mock_set_log):
        from msit_llm.__main__ import LogitsCompare
        args = argparse.Namespace(
            golden_path=PARENT_DIR, my_path=PARENT_DIR, cosine_similarity=0.999, kl_divergence=0.0001,
            l1_norm=0.01, dtype="fp16", output_dir="output", log_level="info"
        )
        LogitsCompare("logitscmp", "help").handle(args)
        mock_cmp.return_value.process_comparsion.assert_called_once()


class TestGetCmdInstance(unittest.TestCase):
    def test_instance(self):
        from msit_llm.__main__ import get_cmd_instance
        cmd = get_cmd_instance()
        # 兼容BaseCommand的属性名为sub_commands或sub_cmds等情况
        has_sub_commands = hasattr(cmd, "sub_commands") or hasattr(cmd, "sub_cmds")
        self.assertFalse(has_sub_commands)
        self.assertTrue(hasattr(cmd, "name"))
        self.assertTrue(hasattr(cmd, "help_info"))
        # 检查子命令数量
        sub_commands = getattr(cmd, "sub_commands", None) or getattr(cmd, "sub_cmds", None)
        self.assertIsNone(sub_commands)


class TestDumpCommandArgs(unittest.TestCase):
    def test_add_arguments(self):
        from msit_llm.__main__ import DumpCommand
        parser = MagicMock()
        DumpCommand("dump", "help").add_arguments(parser)
        # 检查parser.add_argument被多次调用
        self.assertTrue(parser.add_argument.call_count > 0)


class TestCompareCommandArgs(unittest.TestCase):
    def test_add_arguments(self):
        from msit_llm.__main__ import CompareCommand
        parser = MagicMock()
        CompareCommand("compare", "help").add_arguments(parser)
        self.assertTrue(parser.add_argument.call_count > 0)


class TestErrCheckArgs(unittest.TestCase):
    def test_add_arguments(self):
        from msit_llm.__main__ import ErrCheck
        parser = MagicMock()
        ErrCheck("errcheck", "help").add_arguments(parser)
        self.assertTrue(parser.add_argument.call_count > 0)


class TestBCAnalyzeArgs(unittest.TestCase):
    def test_add_arguments(self):
        from msit_llm.__main__ import BCAnalyze
        parser = MagicMock()
        BCAnalyze("analyze", "help").add_arguments(parser)
        self.assertTrue(parser.add_argument.call_count > 0)


class TestBadCaseAnalyzeArgs(unittest.TestCase):
    def test_add_arguments(self):
        from msit_llm.__main__ import BadCaseAnalyze
        parser = MagicMock()
        BadCaseAnalyze("bcanalyze", "help").add_arguments(parser)
        self.assertTrue(parser.add_argument.call_count > 0)


class TestLogitsDumpArgs(unittest.TestCase):
    def test_add_arguments(self):
        from msit_llm.__main__ import LogitsDump
        parser = MagicMock()
        LogitsDump("logitsdump", "help").add_arguments(parser)
        self.assertTrue(parser.add_argument.call_count > 0)


class TestLogitsCompareArgs(unittest.TestCase):
    def test_add_arguments(self):
        from msit_llm.__main__ import LogitsCompare
        parser = MagicMock()
        LogitsCompare("logitscmp", "help").add_arguments(parser)
        self.assertTrue(parser.add_argument.call_count > 0)


class TestTransformCommand(unittest.TestCase):
    def setUp(self):
        import types
        self.backup_modules = {}
        self.modules_to_mock = [
            'torch_npu',
            'tensorflow',
            'tensorflow.compat',
            'tensorflow.compat.v1',
        ]
        for mod in self.modules_to_mock:
            if mod in sys.modules:
                self.backup_modules[mod] = sys.modules[mod]
            mock_mod = MagicMock()
            mock_mod.__spec__ = types.SimpleNamespace()
            sys.modules[mod] = mock_mod
        from msit_llm.__main__ import Transform

    def tearDown(self):
        for mod in self.modules_to_mock:
            if mod in self.backup_modules:
                sys.modules[mod] = self.backup_modules[mod]
            else:
                del sys.modules[mod]
        for mod in list(sys.modules.keys()):
            if mod.startswith("msit_llm.__main__"):
                del sys.modules[mod]

    @patch(f"{MODULE_PATH}.set_log_level")
    @patch(f"{MODULE_PATH}.logger")
    @patch("msit_llm.transform.utils.get_transform_scenario")
    @patch("msit_llm.transform.utils.SCENARIOS")
    def test_handle_torch_to_float_python_atb_with_file(
        self,
        mock_scenarios,
        mock_get_scenario,
        mock_logger,
        mock_set_log
    ):
        # 模拟场景
        mock_get_scenario.return_value = MagicMock()
        mock_scenarios.torch_to_float_python_atb = mock_get_scenario.return_value

        # patch os.path.isfile 返回 True，模拟 quant_disable_names 为文件
        with patch("os.path.isfile", return_value=True), \
             patch("msit_llm.__main__.load_file_to_read_common_check", return_value="dummy.txt"), \
             patch("msit_llm.__main__.ms_open") as mock_ms_open, \
             patch("msit_llm.transform.torch_to_atb_python.transform") as mock_transform:
            mock_file = MagicMock()
            mock_file.readlines.return_value = ["name1\n", "name2\n"]
            mock_ms_open.return_value.__enter__.return_value = mock_file

            args = argparse.Namespace(
                source=PARENT_DIR, to_python=True, to_quant=True, quant_disable_names=FILE_PATH, log_level="info"
            )
            from msit_llm.__main__ import Transform
            Transform("transform", "help").handle(args)
            mock_transform.assert_called_once_with(source_path=PARENT_DIR,
                                                   to_quant=True, quant_disable_names=["name1", "name2"])

    @patch(f"{MODULE_PATH}.set_log_level")
    @patch(f"{MODULE_PATH}.logger")
    @patch("msit_llm.transform.utils.get_transform_scenario")
    @patch("msit_llm.transform.utils.SCENARIOS")
    def test_handle_torch_to_float_python_atb_with_str(
        self,
        mock_scenarios,
        mock_get_scenario,
        mock_logger,
        mock_set_log
    ):
        # 模拟场景
        mock_get_scenario.return_value = MagicMock()
        mock_scenarios.torch_to_float_python_atb = mock_get_scenario.return_value

        # patch os.path.isfile 返回 False，模拟 quant_disable_names 为字符串
        with patch("os.path.isfile", return_value=False), \
             patch("msit_llm.transform.torch_to_atb_python.transform") as mock_transform:
            args = argparse.Namespace(
                source=PARENT_DIR, to_python=True, to_quant=True, quant_disable_names="name1,name2", log_level="info"
            )
            from msit_llm.__main__ import Transform
            Transform("transform", "help").handle(args)
            mock_transform.assert_called_once_with(source_path=PARENT_DIR,
                                                   to_quant=True, quant_disable_names=["name1", "name2"])

    @patch(f"{MODULE_PATH}.set_log_level")
    @patch(f"{MODULE_PATH}.logger")
    @patch("msit_llm.transform.utils.get_transform_scenario")
    @patch("msit_llm.transform.utils.SCENARIOS")
    @patch("msit_llm.transform.float_atb_to_quant_atb.transform_quant.transform_quant")
    def test_handle_float_atb_to_quant_atb(
        self,
        mock_transform_quant,
        mock_scenarios,
        mock_get_scenario,
        mock_logger,
        mock_set_log
    ):
        mock_get_scenario.return_value = MagicMock()
        mock_scenarios.float_atb_to_quant_atb = mock_get_scenario.return_value
        args = argparse.Namespace(
            source=PARENT_DIR, enable_sparse=True, to_python=False, log_level="info"
        )
        from msit_llm.__main__ import Transform
        Transform("transform", "help").handle(args)
        mock_transform_quant.assert_called_once_with(source_path=PARENT_DIR, enable_sparse=True)

    @patch(f"{MODULE_PATH}.set_log_level")
    @patch(f"{MODULE_PATH}.logger")
    @patch("msit_llm.transform.utils.get_transform_scenario")
    @patch("msit_llm.transform.utils.SCENARIOS")
    @patch("msit_llm.transform.torch_to_float_atb.transform_float.transform_report")
    def test_handle_torch_to_float_atb_analyze(
        self,
        mock_transform_report,
        mock_scenarios,
        mock_get_scenario,
        mock_logger,
        mock_set_log
    ):
        mock_get_scenario.return_value = MagicMock()
        mock_scenarios.torch_to_float_atb = mock_get_scenario.return_value
        args = argparse.Namespace(
            source=PARENT_DIR, analyze=True, atb_model_path=PARENT_DIR, to_python=False, log_level="info"
        )
        from msit_llm.__main__ import Transform
        Transform("transform", "help").handle(args)
        mock_transform_report.assert_called_once_with(source_path=PARENT_DIR)

    @patch(f"{MODULE_PATH}.set_log_level")
    @patch(f"{MODULE_PATH}.logger")
    @patch("msit_llm.transform.utils.get_transform_scenario")
    @patch("msit_llm.transform.utils.SCENARIOS")
    @patch("msit_llm.transform.torch_to_float_atb.transform_float.transform_float")
    def test_handle_torch_to_float_atb_transform(
        self,
        mock_transform_float,
        mock_scenarios,
        mock_get_scenario,
        mock_logger,
        mock_set_log
    ):
        mock_get_scenario.return_value = MagicMock()
        mock_scenarios.torch_to_float_atb = mock_get_scenario.return_value
        args = argparse.Namespace(
            source=PARENT_DIR, analyze=False, atb_model_path=PARENT_DIR, to_python=False, log_level="info"
        )
        from msit_llm.__main__ import Transform
        Transform("transform", "help").handle(args)
        mock_transform_float.assert_called_once_with(source_path=PARENT_DIR, atb_model_path=PARENT_DIR)

    @patch(f"{MODULE_PATH}.set_log_level")
    @patch(f"{MODULE_PATH}.logger")
    @patch("msit_llm.transform.utils.get_transform_scenario")
    @patch("msit_llm.transform.utils.SCENARIOS")
    def test_handle_unsupported_scenario(self, mock_scenarios, mock_get_scenario, mock_logger, mock_set_log):
        # 返回一个不属于任何已知场景的对象
        mock_get_scenario.return_value = object()
        mock_scenarios.torch_to_float_python_atb = MagicMock()
        mock_scenarios.float_atb_to_quant_atb = MagicMock()
        mock_scenarios.torch_to_float_atb = MagicMock()

        args = argparse.Namespace(
            source=PARENT_DIR, to_python=False, log_level="info"
        )
        from msit_llm.__main__ import Transform
        with self.assertRaises(ValueError):
            Transform("transform", "help").handle(args)

    def test_add_arguments(self):
        from msit_llm.__main__ import Transform
        parser = MagicMock()
        Transform("transform", "help").add_arguments(parser)
        self.assertTrue(parser.add_argument.call_count > 0)
