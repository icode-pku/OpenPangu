import argparse
from unittest.mock import MagicMock, patch
import pytest
from components.__main__ import main


class TestMainFunction:
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_given_valid_command_and_no_exceptions_when_invoked_then_handle_called_within_umask_context(self, mock_parse_args):
        mock_handle = MagicMock()
        mock_parse_args.return_value = argparse.Namespace(handle=mock_handle)
        main()
        mock_handle.assert_called_once()

    @patch('argparse.ArgumentParser.parse_args')
    def test_main_given_handle_raises_exception_when_invoked_then_exception_wrapped_with_faq_message(self, mock_parse_args):
        mock_handle = MagicMock(side_effect=Exception("Test error"))
        mock_parse_args.return_value = argparse.Namespace(handle=mock_handle)
        with pytest.raises(Exception) as exc_info:
            main()
        assert "Refer FAQ if a known issue" in str(exc_info.value)

    @patch('argparse.ArgumentParser.parse_args')
    def test_main_given_no_handle_attribute_in_args_when_invoked_then_no_action_taken(self, mock_parse_args):
        mock_parse_args.return_value = argparse.Namespace()
        main()  # Should not raise any exception

    @patch('argparse.ArgumentParser.parse_args')
    def test_main_given_invalid_command_and_parser_error_when_parsing_then_parser_error_raised(self, mock_parse_args):
        mock_parse_args.side_effect = SystemExit
        with pytest.raises(SystemExit):
            main()