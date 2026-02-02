import unittest
from unittest.mock import patch, MagicMock, call
import argparse

# msit/components/convert/model_convert/test___main__.py


from model_convert.__main__ import (
    ModelConvertCommand,
    get_cmd_instance
)


class TestModelConvertCommand(unittest.TestCase):
    @patch("model_convert.__main__.add_arguments")
    def test_add_arguments(self, mock_add_args):
        cmd = ModelConvertCommand(backend="atc", name="atc", help_info="help")
        parser = MagicMock()
        cmd.add_arguments(parser)
        mock_add_args.assert_called_once_with(parser, backend="atc")
        self.assertEqual(cmd.conf_args, mock_add_args.return_value)

    @patch("model_convert.__main__.execute_cmd")
    @patch("model_convert.__main__.gen_convert_cmd")
    def test_handle(self, mock_gen_cmd, mock_exec_cmd):
        cmd = ModelConvertCommand(backend="aoe", name="aoe", help_info="help")
        cmd.conf_args = "conf"
        args = MagicMock()
        mock_gen_cmd.return_value = "cmd"
        cmd.handle(args)
        mock_gen_cmd.assert_called_once_with("conf", args, backend="aoe")
        mock_exec_cmd.assert_called_once_with("cmd")


class TestGetCmdInstance(unittest.TestCase):
    def test_get_cmd_instance(self):
        cmd = get_cmd_instance()
        self.assertEqual(cmd.name, "convert")
        self.assertIn("convert tool", cmd.help_info)
        names = [c.name for c in cmd.children]
        self.assertIn("atc", names)
        self.assertIn("aoe", names)

if __name__ == "__main__":
    unittest.main()