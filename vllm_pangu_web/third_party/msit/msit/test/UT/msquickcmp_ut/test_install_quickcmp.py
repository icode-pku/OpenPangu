import unittest
from unittest.mock import patch, MagicMock

from components.debug.compare.msquickcmp.__install__ import CompareInstall


class TestCompareInstall(unittest.TestCase):
    @patch("components.debug.compare.msquickcmp.__install__.pkg_resources.working_set", new_callable=MagicMock)
    @patch("components.debug.compare.msquickcmp.__install__.os.path.exists", new_callable=MagicMock)
    @patch("components.debug.compare.msquickcmp.__install__.os.path.dirname", return_value="/fake/dir")
    def test_check_all_installed_and_file_exists(self, mock_dirname, mock_exists, mock_working_set):
        mock_working_set.__iter__.return_value = [MagicMock(key="ais-bench"), MagicMock(key="msit-surgeon")]
        mock_exists.return_value = True

        result = CompareInstall.check()
        self.assertEqual(result, "OK")

    @patch("components.debug.compare.msquickcmp.__install__.pkg_resources.working_set", new_callable=MagicMock)
    @patch("components.debug.compare.msquickcmp.__install__.os.path.exists", new_callable=MagicMock)
    @patch("components.debug.compare.msquickcmp.__install__.os.path.dirname", return_value="/fake/dir")
    def test_check_missing_packages_and_file(self, mock_dirname, mock_exists, mock_working_set):
        mock_working_set.__iter__.return_value = [MagicMock(key="other-package")]
        mock_exists.return_value = False

        result = CompareInstall.check()
        self.assertIn("msit-benchmark not installed", result)
        self.assertIn("msit-surgeon not installed", result)
        self.assertIn("build lib saveom.so failed", result)

    @patch("components.debug.compare.msquickcmp.__install__.sys.platform", "linux")
    @patch("components.debug.compare.msquickcmp.__install__.subprocess.run")
    @patch("components.debug.compare.msquickcmp.__install__.os.path.abspath", return_value="/fake/dir/install.sh")
    @patch("components.debug.compare.msquickcmp.__install__.os.path.dirname", return_value="/fake/dir")
    def test_build_extra_on_linux(self, mock_dirname, mock_abspath, mock_subproc_run):
        CompareInstall.build_extra()
        mock_subproc_run.assert_called_once_with(["/bin/bash", "/fake/dir/install.sh"])

    @patch("components.debug.compare.msquickcmp.__install__.sys.platform", "win32")
    @patch("components.debug.compare.msquickcmp.__install__.subprocess.run")
    def test_build_extra_on_windows(self, mock_subproc_run):
        CompareInstall.build_extra()
        mock_subproc_run.assert_not_called()
