# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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


import os
import warnings
from datetime import datetime, timezone

import pandas as pd

from components.utils.constants import FileCheckConst
from components.utils.file_utils import check_if_valid_dir_pattern_path, check_and_get_real_path
from msit_llm.common.log import logger
from msit_llm.common.validate import validate_parameters_by_type, validate_parameters_by_func
from components.utils.security_check import ms_makedirs


class PermissionWarning(UserWarning):
    pass


def check_dir(output_dir):
    if not output_dir:
        raise OSError("Output directory should not be empty,")

    if len(output_dir) > 4095 or any(len(item) > 255 for item in output_dir.split(r"/")):
        raise OSError("Output directory should not be too long.")

    if os.path.islink(output_dir):
        warnings.warn("Your attempt to use soft links as directories has triggered a security alert "
                      "and is being monitored closely for potential security threats.", PermissionWarning)

    # will not modify the original output_dir itself
    output_dir = os.path.abspath(output_dir)
    if os.path.isdir(output_dir):
        dir_stat = os.stat(output_dir)
        if dir_stat.st_uid != os.getuid():
            logger.warning("You are attempting to modify a directory that belongs to another entity.")

        if not os.access(output_dir, os.X_OK):
            raise PermissionError(f"{output_dir} is not executable for current user: Permission denied.")

        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"{output_dir} is not writable for current user: Permission denied.")

    else:
        logger.warning("Specified directory does not exist. Trying to create...")
        ms_makedirs(output_dir, mode=0o750, exist_ok=True)


class CaseFilter(object):

    def __init__(self):
        self._metrics = []
        self._columns = ["model input", "model output", "gold standard"]

    def add_metrics(self, **metrics):
        from msit_llm.metrics.metrics import get_metric

        for metric_name, thr in metrics.items():
            self._metrics.append(get_metric(metric_name, thr))
            self._columns.append(metric_name)

    @validate_parameters_by_type(
        {
            "ins": [list],
            "outs": [list],
            "refs": [list],
        },
        in_class=True
    )
    @validate_parameters_by_func(
        {
            "ins": [len, lambda ins: all(isinstance(item, str) for item in ins)],  # should not empty, should be str
            "outs": [len, lambda ins: all(isinstance(item, str) for item in ins)],
            "refs": [len, lambda ins: all(isinstance(item, str) for item in ins)],
        },
        in_class=True
    )
    def apply(self, ins, outs, refs, output_dir=None):
        if not len(ins) == len(outs) == len(refs):
            raise ValueError("Input sequences must have the same length.")

        if not self._metrics:
            raise RuntimeError("Metrics must be added first before calling apply.")

        if output_dir is not None and not isinstance(output_dir, str):
            raise TypeError(f"output_dir should be a string, but got a {type(output_dir)}")

        if output_dir is None:
            output_dir = os.getcwd()
        check_if_valid_dir_pattern_path(output_dir)
        output_dir = check_and_get_real_path(output_dir, FileCheckConst.WRITE_ABLE, must_exist=False)

        data = dict()
        num_metrics = len(self._metrics)

        for col_idx, metric_object in enumerate(self._metrics):
            logger.info("Current metrics: %s", metric_object.__class__.__name__)
            for row_idx, score in metric_object.compare_two_lists_of_words(outs, refs):
                if row_idx not in data:
                    data[row_idx] = [ins[row_idx], outs[row_idx], refs[row_idx]] + [None] * num_metrics

                data[row_idx][col_idx + 3] = score

        self._save(data, output_dir)
        logger.info("Filtering finished.")

    @validate_parameters_by_type(
        {
            "data": [dict],
            "output_dir": [str],
        },
        in_class=True
    )
    @validate_parameters_by_func(
        {
            "data": [],
            "output_dir": [check_dir],
        },
        in_class=True
    )
    def _save(self, data, output_dir):
        if not data:
            logger.warning("Bad cases are unexpectedly empty: No data to save and hence no file will be created. "
                           "If this is normal, please ignore.")
            return

        output_dir = os.path.abspath(output_dir)

        df = pd.DataFrame.from_dict(data, orient="index", columns=self._columns)
        df = df.round(5)
        df.fillna("PASSED", inplace=True)

        time_stamp = datetime.now(tz=timezone.utc).strftime(r"%Y%m%d%H%M%S")
        output_path = os.path.join(output_dir, f"msit_filter_result_{time_stamp}.csv")
        df.to_csv(output_path, encoding='utf-8', index=False)
        os.chmod(output_path, 0o640)
