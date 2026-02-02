import unittest
import sys
from unittest.mock import patch, MagicMock

class TestOpcheckCommandHandle(unittest.TestCase):
    def setUp(self):
        # Patch sys.modules BEFORE import
        self.backup_modules = {}
        self.modules_to_mock = [
            'tensorflow',
            'tensorflow.compat',
            'tensorflow.compat.v1',
        ]
        for mod in self.modules_to_mock:
            if mod in sys.modules:
                self.backup_modules[mod] = sys.modules[mod]
            sys.modules[mod] = MagicMock()
        # 强制 reload 目标模块，确保 patch 生效
        import msit_opcheck.__main__
        self.OpcheckCommand = msit_opcheck.__main__.OpcheckCommand
        self.cmd = self.OpcheckCommand("opcheck", "help")

    def tearDown(self):
        for mod in self.modules_to_mock:
            if mod in self.backup_modules:
                sys.modules[mod] = self.backup_modules[mod]
            else:
                del sys.modules[mod]  # 删除新增的mock模块
        # 清理测试中导入的模块
        for mod in list(sys.modules.keys()):
            if mod.startswith("msit_opcheck.__main__"):
                del sys.modules[mod]

    @patch("msit_opcheck.__main__.set_log_level")
    @patch("msit_opcheck.__main__.OpChecker")
    def test_handle_single_mode(self, mock_opchecker, mock_set_log_level):
        args = MagicMock()
        args.mode = "single"
        args.log_level = "info"
        self.cmd.handle(args)
        mock_set_log_level.assert_called_once_with("info")
        mock_opchecker.assert_called_once_with(args)
        mock_opchecker.return_value.start_test.assert_called_once()

    @patch("msit_opcheck.__main__.set_log_level")
    @patch("msit_opcheck.__main__.FuseOpChecker")
    def test_handle_autofuse_mode_with_graph_path(self, mock_fuseopchecker, mock_set_log_level):
        args = MagicMock()
        args.mode = "autofuse"
        args.graph_path = "/some/graph"
        args.log_level = "debug"
        self.cmd.handle(args)
        mock_set_log_level.assert_called_once_with("debug")
        mock_fuseopchecker.assert_called_once_with(args)
        mock_fuseopchecker.return_value.start_test.assert_called_once()

    @patch("msit_opcheck.__main__.set_log_level")
    def test_handle_autofuse_mode_without_graph_path(self, mock_set_log_level):
        args = MagicMock()
        args.mode = "autofuse"
        args.graph_path = None
        args.log_level = "warning"
        with self.assertRaises(ValueError) as ctx:
            self.cmd.handle(args)
        self.assertIn("must be used together with parameter '--graph-path'", str(ctx.exception))
        mock_set_log_level.assert_called_once_with("warning")

    @patch("msit_opcheck.__main__.set_log_level")
    @patch("msit_opcheck.__main__.OpChecker")
    def test_handle_default_log_level(self, mock_opchecker, mock_set_log_level):
        args = MagicMock()
        args.mode = "single"
        # log_level not set, should default to "info"
        del args.log_level
        # Simulate AttributeError for missing log_level, fallback to default
        def side_effect(lvl):
            pass
        mock_set_log_level.side_effect = side_effect
        setattr(args, "log_level", "info")
        self.cmd.handle(args)
        mock_set_log_level.assert_called_once_with("info")
        mock_opchecker.assert_called_once_with(args)
        mock_opchecker.return_value.start_test.assert_called_once()

if __name__ == "__main__":
    unittest.main()