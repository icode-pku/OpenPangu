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

import os
import shutil
import unittest
from pathlib import Path
from argparse import Namespace
import numpy as np
import pandas as pd

from msserviceprofiler.ms_service_profiler_ext.exporters.exporter_prefill import ExporterPrefill


class TestExporterPrefillFunction(unittest.TestCase):
    def setUp(self):
        self.data_mindie = {"tx_data_df": self.mock_data_mindie()}
        self.data_vllm = {"tx_data_df": self.mock_data_vllm()}

    def mock_data_mindie(self):
        name_list = [
            "encode",
            "httpReq",
            "batchFrameworkProcessing",
            "serializeExcueteMessage",
            "setInferBuffer",
            "grpcWriteToSlave",
            "deserializeExecuteRequestsForInfer",
            "convertTensorBatchToBackend",
            "getInputMetadata",
            "preprocess",
            "forward",
            "sample",
            "postprocess",
            "generateOutput",
            "processPythonExecResult",
            "deserializeExecuteResponse",
            "httpRes",
        ]
        data = {
            "name": name_list * 3,
        }
        len_data = len(data["name"])
        data["during_time"] = np.random.rand(len_data) * 1000
        data["pid"] = ["40"] * len_data
        data["tid"] = ["100"] * len_data
        data["start_time"] = np.arange(len_data) * 1000
        data["end_time"] = np.arange(len_data) * 1000 + 50
        data["start_datetime"] = ["123"] * len_data
        data["end_datetime"] = ["123"] * len_data
        data["batch_type"] = [""] * len_data
        data["rid"] = [""] * len_data
        data["batch_size"] = [None] * len_data
        data["rid_list"] = [None] * len_data
        data["token_id_list"] = [["1234", "5678"]] * len_data
        batch_indices = np.where(np.array(data["name"]) == "batchFrameworkProcessing")[0]
        forward_indices = np.where(np.array(data["name"]) == "forward")[0]
        rid_list = []
        for i, num in enumerate(batch_indices):
            data["batch_type"][num] = "Prefill"
            data["batch_size"][num] = "4"
            data["rid"][num] = str(i + 1)  # 递增的 rid 值
            rid_list.append(str(i + 1))
            data["rid_list"][num] = [str(i + 1)]
        http_req_indices = np.where(np.array(data["name"]) == "httpReq")[0]
        cur = 0
        for i in http_req_indices:
            data["rid"][i] = rid_list[cur]
            cur += 1

        for i, num in enumerate(forward_indices):
            data["rid"][num] = str(i + 1)
            data["rid_list"][num] = [str(i + 1)]

        serialize_indices = np.where(np.array(data["name"]) == "serializeExcueteMessage")[0]
        for i in serialize_indices:
            # 找到前一个 batchFrameworkProcessing 的 end_time
            previous_batch_index = np.where(np.array(data["name"])[:i] == "batchFrameworkProcessing")[0]
            if len(previous_batch_index) > 0:
                previous_end_time = data["end_time"][previous_batch_index[-1]]
                # 确保 start_time 与前一个 end_time 之间的差值小于 100000
                data["start_time"][i] = min(data["start_time"][i], previous_end_time + 100000)
        df = pd.DataFrame(data)
        return df

    def mock_data_vllm(self):
        name_list = [
            "encode",
            "httpReq",
            "batchFrameworkProcessing",
            "preprocess",
            "forward",
            "httpRes"
        ]
        data = {
            "name": name_list * 3,
        }
        len_data = len(data["name"])
        data["during_time"] = np.random.rand(len_data) * 1000
        data["pid"] = ["40"] * len_data
        data["tid"] = ["100"] * len_data
        data["start_time"] = np.arange(len_data) * 1000
        data["end_time"] = np.arange(len_data) * 1000 + 50
        data["start_datetime"] = ["123"] * len_data
        data["end_datetime"] = ["123"] * len_data
        data["batch_type"] = [""] * len_data
        data["rid"] = [""] * len_data
        data["batch_size"] = [None] * len_data
        data["rid_list"] = [None] * len_data
        data["token_id_list"] = [["1234", "5678"]] * len_data
        batch_indices = np.where(np.array(data["name"]) == "batchFrameworkProcessing")[0]
        forward_indices = np.where(np.array(data["name"]) == "forward")[0]
        rid_list = []
        for i, num in enumerate(batch_indices):
            data["batch_type"][num] = "Prefill"
            data["batch_size"][num] = "4"
            data["rid"][num] = str(i + 1)  # 递增的 rid 值
            rid_list.append(str(i + 1))
            data["rid_list"][num] = [str(i + 1)]

        for i, num in enumerate(forward_indices):
            data["rid"][num] = str(i + 1)
            data["rid_list"][num] = [str(i + 1)]
        http_req_indices = np.where(np.array(data["name"]) == "httpReq")[0]
        cur = 0
        for _, num in enumerate(http_req_indices):
            data["rid"][num] = rid_list[cur]
            cur += 1

        df = pd.DataFrame(data)
        return df

    def test_exporter_prefill(self):
        args = Namespace(
            output_path=os.path.join(os.getcwd(), "output"),
            log_level="debug",
            prefill_batch_size=4,
            prefill_number=1,
            prefill_rid="-1",
        )
        try:
            os.makedirs(args.output_path, exist_ok=True)
            os.chmod(args.output_path, 0o740)
            file_path = Path(args.output_path, "prefill.csv")
            ExporterPrefill.initialize(args)
            ExporterPrefill.export(self.data_mindie)
            self.assertTrue(file_path.is_file())
        finally:
            shutil.rmtree(args.output_path)

        try:
            os.makedirs(args.output_path, exist_ok=True)
            os.chmod(args.output_path, 0o740)
            file_path = Path(args.output_path, "prefill.csv")
            ExporterPrefill.initialize(args)
            ExporterPrefill.export(self.data_vllm)
            self.assertTrue(file_path.is_file())
        finally:
            shutil.rmtree(args.output_path)

    def test_exporter_prefill_rid(self):
        args = Namespace(
            output_path=os.path.join(os.getcwd(), "output"),
            log_level="debug",
            prefill_batch_size=4,
            prefill_number=1,
            prefill_rid="1",
        )
        try:
            os.makedirs(args.output_path, exist_ok=True)
            os.chmod(args.output_path, 0o740)
            file_path = Path(args.output_path, "prefill.csv")
            ExporterPrefill.initialize(args)
            ExporterPrefill.export(self.data_mindie)
            self.assertTrue(file_path.is_file())
        finally:
            shutil.rmtree(args.output_path)

        try:
            os.makedirs(args.output_path, exist_ok=True)
            os.chmod(args.output_path, 0o740)
            file_path = Path(args.output_path, "prefill.csv")
            ExporterPrefill.initialize(args)
            ExporterPrefill.export(self.data_vllm)
            self.assertTrue(file_path.is_file())
        finally:
            shutil.rmtree(args.output_path)