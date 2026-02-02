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

import os
import sys
from unittest.mock import patch, MagicMock
import pytest

from components.utils.install import is_windows, warning_in_windows, get_base_path, get_real_pkg_path, AitInstaller, \
    INSTALL_INFO_MAP, install_tools, build_extra, download_comps, get_entry_points


logger = MagicMock()
subprocess = MagicMock()

@pytest.fixture(autouse=True)
def setup():
    global logger, subprocess
    with patch('components.utils.install.logger', logger), \
         patch('components.utils.install.subprocess', subprocess):
        yield


@pytest.mark.filterwarnings("ignore:Deprecated call to:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:pkg_resources is deprecated:DeprecationWarning")
@patch('importlib.metadata.entry_points')
@patch('pkg_resources.iter_entry_points')
def test_get_entry_points_given_valid_entry_points_name_and_importlib_metadata_throws_exception_when_fallback_to_pkg_resources_then_entry_points_returned(mock_iter_entry_points, mock_entry_points):
    mock_entry_points.side_effect = Exception('Test exception')
    mock_iter_entry_points.return_value = ['entry_point1', 'entry_point2']
    result = get_entry_points('test_entry_points')
    assert result == ['entry_point1', 'entry_point2']
    mock_iter_entry_points.assert_called_once_with('test_entry_points')


def test_is_windows_given_windows_platform_when_called_then_true():
    with patch.object(sys, 'platform', 'win32'):
        assert is_windows() is True


def test_is_windows_given_non_windows_platform_when_called_then_false():
    with patch.object(sys, 'platform', 'linux'):
        assert is_windows() is False


def test_warning_in_windows_given_windows_and_unsupported_package_when_called_then_log_warning():
    title = "some package"
    with patch.object(sys, 'platform', 'win32'), \
         patch('components.utils.install.is_windows', return_value=True):
        result = warning_in_windows(title)
        logger.warning.assert_called_once_with(f"{title} is not support windows")
        assert result is True


def test_get_base_path_when_called_then_return_absolute_path():
    expected_path = "/absolute/path/to/base"
    with patch('os.path.abspath') as mock_abspath:
        mock_abspath.return_value = expected_path
        path = get_base_path()
        logger.info.assert_called_once_with(expected_path)
        assert path == expected_path


def test_get_real_pkg_path_given_relative_path_when_called_then_joined_with_base_path():
    relative_path = "relative/path"
    base_path = "/base"
    expected_path = os.path.join(base_path, relative_path)
    with patch('components.utils.install.get_base_path', return_value=base_path):
        path = get_real_pkg_path(relative_path)
        assert path == expected_path


def test_ait_installer_check_when_called_then_return_ok():
    result = AitInstaller.check()
    assert result == "OK"


def test_install_tools_given_all_when_called_then_install_all_packages():
    names = ["all"]
    find_links = None
    with patch('components.utils.install.install_tool') as mock_install_tool:
        install_tools(names, find_links)
        assert mock_install_tool.call_count == len(INSTALL_INFO_MAP)


def test_build_extra_given_component_name_when_called_then_log_building_extra():
    name = "graph"
    find_links = "find_links_dir"
    with patch('components.utils.install.get_installer', return_value=AitInstaller()) as mock_get_installer, \
         patch('components.utils.install.logger') as mock_logger:
        build_extra(name, find_links)
        mock_logger.info.assert_any_call("building extra of msit-graph")


def test_download_comps_given_all_when_called_then_download_all_packages():
    names = ["all"]
    dest = "dest_dir"
    with patch('components.utils.install.download_comp') as mock_download_comp:
        download_comps(names, dest)
        assert mock_download_comp.call_count == len(INSTALL_INFO_MAP)
