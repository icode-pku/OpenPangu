# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

import datetime
import logging
import os
import subprocess
import sys
import threading

stop_print_thread = False


def print_stout(output):
    while True:
        line = output.readline().strip()
        if line:
            logging.info(line)
        global stop_print_thread
        if stop_print_thread:
            break


def stop_stout_threads(thread_list):
    global stop_print_thread
    stop_print_thread = True
    for stout_thread in thread_list:
        if stout_thread.is_alive():
            stout_thread.join()


def start_st_process(module_name):
    st_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "st", module_name)
    cmd = ["python3", "-m", "pytest", "-s", st_path]
    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stout_thread = threading.Thread(target=print_stout, args=(process.stdout,))
    stout_thread.start()
    return process, stout_thread


def stop_st_process(process_list):
    for process in process_list:
        if process.poll() is None:
            process.terminate()
            process.wait()


def run_st(module_choice):
    timeout = 3600
    modules = ["analyze", "split"]
    if module_choice != "all":
        modules = [module_choice]
    process_list = []
    thread_list = []
    for module in modules:
        process, stout_thread = start_st_process(module)
        process_list.append(process)
        thread_list.append(stout_thread)

    success, failed = True, False
    start_time = datetime.datetime.utcnow()
    while process_list:
        duration = datetime.datetime.utcnow() - start_time
        if duration.total_seconds() >= timeout:
            logging.error("run st use case timeout.")
            stop_stout_threads(thread_list)
            stop_st_process(process_list)
            return failed
        for process in process_list:
            if process.poll() is None:
                continue
            if process.returncode == 0:
                process_list.remove(process)
                continue
            stop_stout_threads(thread_list)
            stop_st_process(process_list)
            return failed
    stop_stout_threads(thread_list)
    return success


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if '--module' in sys.argv:
        module_index = sys.argv.index("--module") + 1
        module = sys.argv[module_index]
    else:
        module = "all"
    st_success = run_st(module)
    if st_success:
        logging.info("run st successfully.")
        sys.exit(0)
    else:
        logging.error("run st failed.")
        sys.exit(1)