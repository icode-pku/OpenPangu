# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from unittest.mock import MagicMock, patch
import pytest

from components.utils.parser import BaseCommand, LazyEntryPointCommand, DownloadCommand, \
                                    AitInstallCommand, AitCheckCommand, AitBuildExtraCommand, \
                                    ALL_SUB_TOOLS_WITH_ALL, ALL_SUB_TOOLS



@pytest.fixture
def mock_entry_point():
    entry_point = MagicMock()
    entry_point.name = "mock_entry:Mock help"
    entry_point.load.return_value = BaseCommand("test", "Test command")
    return entry_point


def test_base_command_register_parser_given_parser_when_valid_then_pass():
    # Mock argparse.ArgumentParser
    parser = MagicMock(spec=argparse.ArgumentParser)
    command = BaseCommand("test_command", "Test command description")

    # Mock add_arguments method
    command.add_arguments = MagicMock()

    command.register_parser(parser)

    command.add_arguments.assert_called_once()
    parser.set_defaults.assert_called_once()


def test_base_command_register_parser_given_parser_when_children_then_subparsers_created():
    parser = MagicMock(spec=argparse.ArgumentParser)
    child_command = BaseCommand("child_command", "Child command description")
    command = BaseCommand("test_command", "Test command description", children=[child_command])

    command.register_parser(parser)

    # Check subparsers are created
    parser.add_subparsers.assert_called_once()


def test_base_command_handle_given_parser_when_valid_then_pass():
    parser = MagicMock(spec=argparse.ArgumentParser)
    command = BaseCommand("test_command", "Test command description")

    command.register_parser(parser)
    command.handle(None)

    parser.print_help.assert_called_once()


def test_lazy_entry_point_command_register_parser_given_invalid_child_then_raise_value_error():
    parser = MagicMock(spec=argparse.ArgumentParser)
    with pytest.raises(ValueError):
        command = BaseCommand("test_command", "Test command description", children=[123])  # Invalid child

def test_lazy_entry_point_command_build_lazy_tasks_given_entry_points_name_when_valid_then_returns_tasks(mock_entry_point):
    with patch("components.utils.parser.get_entry_points", return_value=[mock_entry_point]):
        tasks = LazyEntryPointCommand.build_lazy_tasks("group")
        assert len(tasks) > 0
        assert isinstance(tasks[0], LazyEntryPointCommand)

def test_lazy_entry_point_command_build_lazy_tasks_given_entry_points_name_without_colon_then_help_info_empty():
    entry_point = MagicMock()
    entry_point.name = "test"
    with patch("components.utils.parser.get_entry_points", return_value=[entry_point]):
        tasks = LazyEntryPointCommand.build_lazy_tasks("group")
        assert tasks[0].help_info == ""


def test_lazy_entry_point_command_given_parser_and_valid_args_when_hook_parse_args_then_args_parsed_correctly():
    command = LazyEntryPointCommand("test", "help", MagicMock())
    parser = argparse.ArgumentParser()
    command.register_parser(parser)
    args, unknown = parser.parse_known_args(["--test"])
    assert args is not None


def test_ait_install_command_add_arguments_given_parser_when_called_then_add_args_with_choices():
    parser = MagicMock()
    AitInstallCommand().add_arguments(parser)
    parser.add_argument.assert_called()
    for call in parser.add_argument.call_args_list:
        if call[1].get("choices"):
            assert set(call[1]["choices"]) == set(ALL_SUB_TOOLS_WITH_ALL)


def test_ait_install_command_handle_given_args_when_valid_then_call_install_tools():
    with patch("components.utils.parser.install_tools") as mock_install:
        args = MagicMock(comp_names=["tool1"], find_links="path")
        AitInstallCommand().handle(args)
        mock_install.assert_called_with(["tool1"], "path")


def test_ait_check_command_init_given_default_when_created_then_proper_values():
    acc = AitCheckCommand()
    assert acc.name == "check"
    assert acc.help_info == "check msit tools status."


def test_ait_check_command_add_arguments_given_parser_when_called_then_add_args_with_choices():
    parser = MagicMock()
    AitCheckCommand().add_arguments(parser)
    parser.add_argument.assert_called()
    for call in parser.add_argument.call_args_list:
        if call[1].get("choices"):
            assert set(call[1]["choices"]) == set(ALL_SUB_TOOLS_WITH_ALL)

def test_ait_check_command_handle_given_args_when_valid_then_call_check_tools():
    with patch("components.utils.parser.check_tools") as mock_check:
        args = MagicMock(comp_names=["tool1"])
        AitCheckCommand().handle(args)
        mock_check.assert_called_with(["tool1"])

def test_ait_build_extra_command_init_given_default_when_created_then_proper_values():
    abec = AitBuildExtraCommand()
    assert abec.name == "build-extra"
    assert abec.help_info == "build msit tools extra"

def test_ait_build_extra_command_add_arguments_given_parser_when_called_then_add_args_with_choices():
    parser = MagicMock()
    AitBuildExtraCommand().add_arguments(parser)
    parser.add_argument.assert_called()
    for call in parser.add_argument.call_args_list:
        if call[1].get("choices"):
            assert set(call[1]["choices"]) == set(ALL_SUB_TOOLS)

def test_ait_build_extra_command_handle_given_args_when_valid_then_call_build_extra():
    with patch("components.utils.parser.build_extra") as mock_build:
        args = MagicMock(comp_name="tool1", find_links="path")
        AitBuildExtraCommand().handle(args)
        mock_build.assert_called_with("tool1", "path")

def test_download_command_init_given_default_when_created_then_proper_values():
    dc = DownloadCommand()
    assert dc.name == "download"
    assert dc.help_info == "download packages"

def test_download_command_add_arguments_given_parser_when_called_then_add_args_with_choices_and_required_dest():
    parser = MagicMock()
    DownloadCommand().add_arguments(parser)
    parser.add_argument.assert_called()
    for call in parser.add_argument.call_args_list:
        if call[1].get("choices"):
            assert set(call[1]["choices"]) == set(ALL_SUB_TOOLS_WITH_ALL)
        if call[0] == ("--dest", "-d"):
            assert call[1]["required"] == True

def test_download_command_handle_given_args_when_valid_then_call_download_comps():
    with patch("components.utils.parser.download_comps") as mock_download:
        args = MagicMock(comp_names=["tool1"], dest="path")
        DownloadCommand().handle(args)
        mock_download.assert_called_with(["tool1"], "path")
