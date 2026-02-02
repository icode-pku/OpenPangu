# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import subprocess
import os
import sys
import time
from typing import Optional, Union, List
from queue import Queue
import queue
import threading
import select


class CommandExecutor:
    def __init__(self):
        self.process = None
        self._exit_code = None
        self.msg_out_queue = Queue()
        self.inst_in_queue = Queue()
        self.thread = None
        self.env = dict()


    def __del__(self):
        """析构函数，确保清理资源"""
        self._reset()


    def execute(self, command, env=None) -> None:
        """执行已设置的命令"""
        if command is None:
            raise ValueError("No command has been set. Use set_command() first.")

        self._reset()

        logging.info(command)

        sub_process_env = os.environ.copy()
        if env:
            sub_process_env.update(env)

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=sub_process_env,
            universal_newlines=True,
            shell=isinstance(command, str),
        )
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()


    def clean_msg_out_queue(self):
        while not self.msg_out_queue.empty():
            try:
                self.msg_out_queue.get_nowait()
            except queue.Empty:
                break


    def kill(self) -> None:
        """重置执行状态"""
        if self.process is not None:
            subprocess.run(["/usr/bin/pkill", "-P", f"{self.process.pid}"])
            subprocess.run(["/usr/bin/kill", "-9", f"{self.process.pid}"])
        self.process = None
        self._exit_code = None


    def wait(self, target: Optional[str] = None, timeout: Optional[float] = None) -> tuple:
        """
        等待命令执行完成或输出中出现特定字符串

        参数:
            target: 要等待的目标字符串(None表示等待命令结束)
            from_stderr: 是否从标准错误流中查找
            timeout: 超时时间(秒)

        返回:
            错误码，
            条件是否满足
                -1: 条件不满足
                0: 条件满足(找到目标或正常结束)
                1: 超时
            其他: 进程退出码(未找到目标或异常结束)
        """
        if self.process is None:
            raise ValueError("Command has not been executed yet. Call execute() first.")

        start_time = time.time()
        if target is not None:
            self.inst_in_queue.put("get_output")
        if self.process.poll() is not None:
            return self.process.poll(), -1

        while True:
            try:
                output = self.msg_out_queue.get(timeout=1)
                if output is None:
                    return self.msg_out_queue.get(), -1
                elif target in output:
                    self.inst_in_queue.put("not_get_output")
                    return None, 0
                else:
                    pass
            except queue.Empty:
                time.sleep(0.1)

            # 检查超时
            if timeout is not None and (time.time() - start_time) > timeout:
                return None, 1


    def _monitor(self):
        is_get_output = False
        process = self.process

        def read_instruction():
            nonlocal is_get_output
            if self.inst_in_queue.empty():
                return False
            instruction = self.inst_in_queue.get()
            if instruction == "get_output":
                is_get_output = True
            elif instruction == "not_get_output":
                if is_get_output:
                    self.clean_msg_out_queue()
                is_get_output = False
            elif instruction == "exit":
                return True
            else:
                return False
            return False

        while True:
            if read_instruction():
                break
            # 非阻塞检查管道
            reads = [process.stdout, process.stderr]
            ready, _, _ = select.select(reads, [], [], 0.1)

            for stream in ready:
                line = stream.readline()
                if not line:  # 进程结束
                    continue

                # 实时输出
                if stream == process.stdout:
                    logging.info(line)
                else:
                    logging.info(line)

                if is_get_output:
                    self.msg_out_queue.put(line)

            if process.poll() is not None:
                self.msg_out_queue.put(None)
                self.msg_out_queue.put(process.poll())
                break


    def _reset(self) -> None:
        """重置执行状态"""
        if self.process is not None and self._exit_code is None:
            self.process.terminate()
        self.process = None
        self._exit_code = None