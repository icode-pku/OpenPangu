# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

from msserviceprofiler.modelevalstate.config.config import CommunicationConfig, get_settings, \
    map_param_with_value, default_support_field
from msserviceprofiler.modelevalstate.optimizer.server import Scheduler
from msserviceprofiler.modelevalstate.optimizer.communication import CommunicationForFile, CustomCommand


def test_scheduler_init(tmpdir):
    # 创建一个CommunicationConfig对象
    work_path = Path(tmpdir)
    res_file = work_path.joinpath("res.txt")
    cmd_file = work_path.joinpath("cmd.txt")
    communication_config = CommunicationConfig(cmd_file=res_file, res_file=cmd_file)

    # 创建Scheduler对象
    scheduler = Scheduler(communication_config)

    # 检查communication_config是否被正确传递
    assert scheduler.communication_config == communication_config

    # 检查simulator是否被初始化为None
    assert scheduler.simulator is None

    # 检查communication是否被正确初始化
    assert isinstance(scheduler.communication, CommunicationForFile)
    assert scheduler.communication.res_file == res_file
    assert scheduler.communication.cmd_file == cmd_file

    # 检查cmd是否被正确初始化
    assert isinstance(scheduler.cmd, CustomCommand)


def test_backup_path_exists():
    # Arrange
    scheduler = Scheduler(get_settings().communication)
    scheduler.communication = Mock()
    params = "/existing/path"
    _cmd = scheduler.cmd.start
    scheduler.cmd.history = _cmd
    with patch("msserviceprofiler.modelevalstate.optimizer.server.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        scheduler.backup(params)

    # Assert
    scheduler.communication.send_command.assert_called_once_with(f"{scheduler.cmd.history[-1]}:done")
    scheduler.communication.clear_res.assert_called_once()


def test_backup_path_not_exists():
    # Arrange
    scheduler = Scheduler(get_settings().communication)
    scheduler.communication = Mock()
    params = "/non/existing/path"
    _cmd = scheduler.cmd.start
    scheduler.cmd.history = _cmd
    with patch("msserviceprofiler.modelevalstate.optimizer.server.Path") as mock_path:
        mock_path.return_value.exists.return_value = False
        mock_path.return_value.mkdir.return_value = None
        scheduler.backup(params)

    # Assert
    scheduler.communication.send_command.assert_called_once_with(f"{scheduler.cmd.history[-1]}:done")
    scheduler.communication.clear_res.assert_called_once()


def test_check_success_no_simulator():
    scheduler = Scheduler(get_settings().communication)
    scheduler.simulator = None
    assert scheduler.check_success() is None


def test_check_success_simulator_succeeds_immediately():
    scheduler = Scheduler(get_settings().communication)
    scheduler.simulator = Mock()
    scheduler.cmd.history = "check success 1111111"
    scheduler.simulator.check_success.return_value = True
    scheduler.communication = Mock()
    scheduler.check_success()
    scheduler.communication.send_command.assert_called_once_with("check success 1111111:True")


@patch("time.sleep")
def test_check_success_simulator_succeeds_after_retries(mock_sleep):
    scheduler = Scheduler(get_settings().communication)
    scheduler.simulator = Mock()
    scheduler.cmd.history = "check success 1111111"
    scheduler.simulator.check_success.side_effect = [False, False, True]
    scheduler.communication = Mock()
    scheduler.check_success()
    assert scheduler.simulator.check_success.call_count == 3
    scheduler.communication.send_command.assert_called_once_with("check success 1111111:True")
    mock_sleep.call_count == 2


@patch("time.sleep")
def test_check_success_simulator_always_fails(mock_sleep):
    scheduler = Scheduler(get_settings().communication)
    scheduler.simulator = Mock()
    scheduler.cmd.history = "check success 1111111"
    scheduler.simulator.check_success.return_value = False
    scheduler.communication = Mock()
    scheduler.check_success()
    assert scheduler.simulator.check_success.call_count == 10
    scheduler.communication.send_command.assert_called_once_with("check success 1111111:False")
    mock_sleep.call_count == 10


def test_stop_with_simulator():
    # 创建Scheduler实例
    scheduler = Scheduler(get_settings().communication)
    # 模拟simulator和communication对象
    scheduler.simulator = Mock()
    scheduler.communication = Mock()
    scheduler.cmd.history = "stop 1111111 params:True"
    # 测试参数
    params = "True"

    # 调用stop方法
    scheduler.stop(params)

    # 验证simulator.stop被调用
    scheduler.simulator.stop.assert_called_once_with(True)

    # 验证communication.send_command被调用
    scheduler.communication.send_command.assert_called_once()

    # 验证communication.clear_res被调用
    scheduler.communication.clear_res.assert_called_once()


def test_stop_without_simulator():
    # 创建Scheduler实例
    scheduler = Scheduler(get_settings().communication)

    # 设置simulator为None
    scheduler.simulator = None

    # 模拟communication对象
    scheduler.communication = Mock()
    scheduler.cmd.history = "stop 1111111 params:True"

    # 测试参数
    params = "True"

    # 调用stop方法
    scheduler.stop(params)

    # 验证communication.send_command没有被调用
    scheduler.communication.send_command.assert_not_called()

    # 验证communication.clear_res没有被调用
    scheduler.communication.clear_res.assert_not_called()


def test_get_cmd_param_empty():
    scheduler = Scheduler(get_settings().communication)
    scheduler.communication = MagicMock()
    scheduler.communication.recv_command.return_value = ""
    assert scheduler.get_cmd_param() == (None, None)


def test_get_cmd_param_eof():
    scheduler = Scheduler(get_settings().communication)

    scheduler.communication = MagicMock()
    scheduler.communication.recv_command.return_value = "EOF"
    assert scheduler.get_cmd_param() == (None, None)


def test_get_cmd_param_history():
    scheduler = Scheduler(get_settings().communication)
    scheduler.communication = MagicMock()
    scheduler.communication.recv_command.return_value = "cmd1"
    scheduler.cmd.history = ["cmd1"]
    assert scheduler.get_cmd_param() == (None, None)


def test_get_cmd_param_format_error():
    scheduler = Scheduler(get_settings().communication)
    scheduler.communication = MagicMock()
    scheduler.communication.recv_command.return_value = "cmd1"
    assert scheduler.get_cmd_param() == (None, None)


def test_get_cmd_param_success():
    scheduler = Scheduler(get_settings().communication)
    scheduler.communication = MagicMock()
    scheduler.communication.recv_command.return_value = "cmd1 params:123"
    assert scheduler.get_cmd_param() == ("cmd1", "123")


class TestSchedulerProcessPoll:
    @classmethod
    def test_process_poll_with_simulator(cls, scheduler):
        # 模拟simulator.process.poll()返回值
        scheduler.simulator.process.poll.return_value = True
        scheduler.cmd.history = "process_poll 1111111"
        # 调用process_poll方法
        scheduler.process_poll()

        # 验证是否正确调用了相关方法
        scheduler.simulator.process.poll.assert_called_once()
        scheduler.communication.send_command.assert_called_once_with("process_poll 1111111:True")
        scheduler.communication.clear_res.assert_called_once()

    @classmethod
    def test_process_poll_without_simulator(cls, scheduler):
        # 设置simulator为None
        scheduler.simulator = None
        scheduler.cmd.history = "process_poll 1111111"
        # 调用process_poll方法
        scheduler.process_poll()
        # 验证是否正确调用了相关方法
        scheduler.communication.send_command.assert_called_once_with("process_poll 1111111:None")
        scheduler.communication.clear_res.assert_called_once()

    @pytest.fixture
    def scheduler(self):
        # 创建Scheduler实例
        scheduler = Scheduler(get_settings().communication)
        scheduler.simulator = MagicMock()
        scheduler.communication = MagicMock()
        return scheduler


# 测试用例1: 测试当get_cmd_param返回的_cmd为None时，init方法返回False
def test_init_cmd_none():
    scheduler = Scheduler(get_settings().communication)
    scheduler.get_cmd_param = MagicMock(return_value=(None, None))
    assert scheduler.init() is False


# 测试用例2: 测试当get_cmd_param返回的_cmd为"init"时，init方法返回True
def test_init_cmd_init():
    scheduler = Scheduler(get_settings().communication)
    scheduler.get_cmd_param = MagicMock(return_value=("init", None))
    scheduler.cmd.history = "init 11111111"
    scheduler.communication = MagicMock()
    assert scheduler.init() is True
    scheduler.communication.send_command.assert_called_once_with("init 11111111:done")
    scheduler.communication.clear_res.assert_called_once()


# 测试用例3: 测试当get_cmd_param返回的_cmd不为"init"时，init方法返回False
def test_init_cmd_not_init():
    scheduler = Scheduler(get_settings().communication)
    scheduler.get_cmd_param = MagicMock(return_value=("other_command", None))
    assert scheduler.init() is False


def test_run_no_cmd():
    scheduler = Scheduler(get_settings().communication)
    with patch.object(scheduler, 'get_cmd_param', return_value=(None, None)):
        assert scheduler.run() == ''


def test_run_unknown_cmd():
    scheduler = Scheduler(get_settings().communication)
    with patch.object(scheduler, 'get_cmd_param', return_value=('unknown_cmd', None)):
        with patch('msserviceprofiler.modelevalstate.optimizer.server.logger') as mock_logger:
            assert scheduler.run() == ''
            mock_logger.error.assert_called_once()


def test_run_no_param():
    scheduler = Scheduler(get_settings().communication)
    scheduler.command = MagicMock(return_value='result')
    with patch.object(scheduler, 'get_cmd_param', return_value=('command', None)):
        with patch('msserviceprofiler.modelevalstate.optimizer.server.getattr', return_value=scheduler.command):
            assert scheduler.run() == 'result'
            scheduler.command.assert_called_once()


def test_run_with_param():
    scheduler = Scheduler(get_settings().communication)
    scheduler.command = MagicMock(return_value='result')
    with patch.object(scheduler, 'get_cmd_param', return_value=('command', 'param')):
        with patch('msserviceprofiler.modelevalstate.optimizer.server.getattr', return_value=scheduler.command):
            assert scheduler.run() == 'result'
            scheduler.command.assert_called_once_with('param')
