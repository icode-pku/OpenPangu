import unittest
from unittest.mock import patch, MagicMock, call
import argparse

from msit_graph.__main__ import (
    check_output_path_legality,
    check_input_path_legality,
    StatsCommand,
    StripCommand,
    ExtractCommand,
    FuseCommand,
    InspectCommand,
    get_cmd_instance,
)

class TestCheckOutputPathLegality(unittest.TestCase):
    @patch("msit_graph.__main__.FileStat")
    def test_valid(self, mock_filestat):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = True
        mock_filestat.return_value = mock_stat
        self.assertEqual(check_output_path_legality("outdir"), "outdir")

    @patch("msit_graph.__main__.FileStat", side_effect=FileNotFoundError)
    def test_not_exist(self, mock_filestat):
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_output_path_legality("outdir")
        self.assertIn("does not exist", str(cm.exception))

    @patch("msit_graph.__main__.FileStat", side_effect=PermissionError)
    def test_permission(self, mock_filestat):
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_output_path_legality("outdir")
        self.assertIn("permission denied", str(cm.exception))

    @patch("msit_graph.__main__.FileStat", side_effect=Exception("fail"))
    def test_other_exception(self, mock_filestat):
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_output_path_legality("outdir")
        self.assertIn("an unexpected error occurred", str(cm.exception))

    @patch("msit_graph.__main__.FileStat")
    def test_not_writable(self, mock_filestat):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = False
        mock_filestat.return_value = mock_stat
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_output_path_legality("outdir")
        self.assertIn("cannot be written", str(cm.exception))

    def test_empty(self):
        self.assertIsNone(check_output_path_legality(None))


class TestCheckInputPathLegality(unittest.TestCase):
    @patch("msit_graph.__main__.FileStat")
    def test_valid(self, mock_filestat):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = True
        mock_filestat.return_value = mock_stat
        self.assertEqual(check_input_path_legality("indir"), "indir")

    @patch("msit_graph.__main__.FileStat", side_effect=FileNotFoundError)
    def test_not_exist(self, mock_filestat):
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_input_path_legality("indir")
        self.assertIn("does not exist", str(cm.exception))

    @patch("msit_graph.__main__.FileStat", side_effect=PermissionError)
    def test_permission(self, mock_filestat):
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_input_path_legality("indir")
        self.assertIn("permission denied", str(cm.exception))

    @patch("msit_graph.__main__.FileStat", side_effect=Exception("fail"))
    def test_other_exception(self, mock_filestat):
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_input_path_legality("indir")
        self.assertIn("an unexpected error occurred", str(cm.exception))

    @patch("msit_graph.__main__.FileStat")
    def test_not_readable(self, mock_filestat):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = False
        mock_filestat.return_value = mock_stat
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_input_path_legality("indir")
        self.assertIn("cannot be read", str(cm.exception))

    def test_empty(self):
        self.assertIsNone(check_input_path_legality(None))


class TestStatsCommand(unittest.TestCase):
    @patch("msit_graph.__main__.GraphAnalyze.print_graph_stat")
    @patch("msit_graph.__main__.set_log_level")
    def test_handle(self, mock_set_log, mock_print_stat):
        cmd = StatsCommand("stats", "help")
        args = argparse.Namespace(input="in.pbtxt", log_level="info")
        cmd.handle(args)
        mock_set_log.assert_called_with("info")
        mock_print_stat.assert_called_with("in.pbtxt")

    @patch("msit_graph.__main__.check_input_path_legality", side_effect=lambda x: x)
    def test_add_arguments(self, mock_check):
        cmd = StatsCommand("stats", "help")
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        args = parser.parse_args(["-i", "in.pbtxt", "-l", "info"])
        self.assertEqual(args.input, "in.pbtxt")
        self.assertEqual(args.log_level, "info")


class TestStripCommand(unittest.TestCase):
    @patch("msit_graph.__main__.GraphAnalyze.strip")
    @patch("msit_graph.__main__.set_log_level")
    def test_handle(self, mock_set_log, mock_strip):
        cmd = StripCommand("strip", "help")
        args = argparse.Namespace(input="in.pbtxt", level=2, output="out.pbtxt", log_level="debug")
        cmd.handle(args)
        mock_set_log.assert_called_with("debug")
        mock_strip.assert_called_with("in.pbtxt", 2, "out.pbtxt")

    @patch("msit_graph.__main__.check_input_path_legality", side_effect=lambda x: x)
    @patch("msit_graph.__main__.check_output_path_legality", side_effect=lambda x: x)
    def test_add_arguments(self, mock_check_out, mock_check_in):
        cmd = StripCommand("strip", "help")
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        args = parser.parse_args(["-i", "in.pbtxt", "--level", "1", "-o", "out.pbtxt", "-l", "info"])
        self.assertEqual(args.input, "in.pbtxt")
        self.assertEqual(args.level, 1)
        self.assertEqual(args.output, "out.pbtxt")
        self.assertEqual(args.log_level, "info")


class TestExtractCommand(unittest.TestCase):
    @patch("msit_graph.__main__.GraphAnalyze.extract_sub_graph")
    @patch("msit_graph.__main__.set_log_level")
    def test_handle(self, mock_set_log, mock_extract):
        cmd = ExtractCommand("extract", "help")
        args = argparse.Namespace(
            input="in.pbtxt", output="out.pbtxt", start_node=None, end_node=None,
            center_node=None, layer_number=1, only_forward=False, only_backward=False,
            without_leaves=False, stop_name=None, log_level="info"
        )
        cmd.handle(args)
        mock_set_log.assert_called_with("info")
        mock_extract.assert_called_with(args)

    @patch("msit_graph.__main__.check_input_path_legality", side_effect=lambda x: x)
    def test_add_arguments(self, mock_check):
        cmd = ExtractCommand("extract", "help")
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        args = parser.parse_args([
            "-i", "in.pbtxt", "-o", "out.pbtxt", "--start-node", "a", "--end-node", "b",
            "--center-node", "c", "--layer-number", "2", "--only-forward", "--only-backward",
            "--without-leaves", "--stop-name", "s1", "--stop-name", "s2", "-l", "info"
        ])
        self.assertEqual(args.input, "in.pbtxt")
        self.assertEqual(args.output, "out.pbtxt")
        self.assertEqual(args.start_node, "a")
        self.assertEqual(args.end_node, "b")
        self.assertEqual(args.center_node, "c")
        self.assertEqual(args.layer_number, 2)
        self.assertTrue(args.only_forward)
        self.assertTrue(args.only_backward)
        self.assertTrue(args.without_leaves)
        self.assertEqual(args.stop_name, ["s1", "s2"])
        self.assertEqual(args.log_level, "info")


class TestFuseCommand(unittest.TestCase):
    @patch("msit_graph.__main__.calculate_sum")
    @patch("msit_graph.__main__.set_log_level")
    def test_handle(self, mock_set_log, mock_calc):
        cmd = FuseCommand("fuse", "help")
        args = argparse.Namespace(
            source="in.pbtxt", profile="profile.csv", max_nodes=8, min_nodes=2, min_times=1,
            output="out.csv", log_level="info"
        )
        cmd.handle(args)
        mock_set_log.assert_called_with("info")
        mock_calc.assert_called_with(args)

    @patch("msit_graph.__main__.check_input_path_legality", side_effect=lambda x: x)
    @patch("msit_graph.__main__.check_output_path_legality", side_effect=lambda x: x)
    def test_add_arguments(self, mock_check_out, mock_check_in):
        cmd = FuseCommand("fuse", "help")
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        args = parser.parse_args([
            "-s", "in.pbtxt", "-p", "profile.csv", "--max-nodes", "10", "--min-nodes", "3",
            "--min-times", "2", "-o", "out.csv", "-l", "info"
        ])
        self.assertEqual(args.source, "in.pbtxt")
        self.assertEqual(args.profile, "profile.csv")
        self.assertEqual(args.max_nodes, 10)
        self.assertEqual(args.min_nodes, 3)
        self.assertEqual(args.min_times, 2)
        self.assertEqual(args.output, "out.csv")
        self.assertEqual(args.log_level, "info")


class TestInspectCommand(unittest.TestCase):
    @patch("msit_graph.__main__.execute")
    @patch("msit_graph.__main__.set_log_level")
    def test_handle(self, mock_set_log, mock_execute):
        cmd = InspectCommand("inspect", "help")
        args = argparse.Namespace(
            input="in.pbtxt", type="dshape", log_level="info", output="./"
        )
        cmd.handle(args)
        mock_set_log.assert_called_with("info")
        mock_execute.assert_called_with(args)

    @patch("msit_graph.__main__.check_input_path_legality", side_effect=lambda x: x)
    @patch("msit_graph.__main__.check_output_path_legality", side_effect=lambda x: x)
    def test_add_arguments(self, mock_check_out, mock_check_in):
        cmd = InspectCommand("inspect", "help")
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        args = parser.parse_args([
            "-i", "in.pbtxt", "-t", "dshape", "-l", "info", "-o", "./"
        ])
        self.assertEqual(args.input, "in.pbtxt")
        self.assertEqual(args.type, "dshape")
        self.assertEqual(args.log_level, "info")
        self.assertEqual(args.output, "./")