# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd. All rights reserved.
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
import importlib
from collections import defaultdict
import numpy as np
import pandas as pd

from components.debug.common import logger
from components.utils.security_check import check_input_path_legality, check_output_path_legality
from components.utils.cmp_algorithm import NP_CMP_ALG_MAP
from msit_opcheck.util.file_read import get_ascbackend_ascgraph, convert_ge_dump_file_to_npy
from msit_opcheck.autofuse.tf_builder import convert_to_tf_graph, sanitize_filename


class FuseOpChecker:
    SUPPORT_OP_LIST = ["AscBackend", "FusedAscBackend"]
    PRECISION_METRIC = ['max_relative_error', 'cosine_similarity', 'kl_divergence']

    def __init__(self, args):
        self.input_path = args.input  # GE dump data path
        self.output_path = args.output
        self.graph_path = args.graph_path  # GE dump graph path
        self.npy_path = None
        self.graph_name_to_input_map = {}
        self.opname_to_dump_data_map = {}
        self.compare_result = defaultdict(list)

    @staticmethod
    def _load_pyautofuse_graph(file_path):
        """从指定的 Python 文件加载 pyautofuse 构建的图"""
        file_path = check_input_path_legality(file_path) # 加载文件前校验一下权限
        spec = importlib.util.spec_from_file_location("graph_module", file_path)
        graph_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graph_module)
        return graph_module

    @staticmethod
    def _run_tfgraph(tf_graph_builder, feed_dict):
        import tensorflow.compat.v1 as tf
        tf_graph = tf_graph_builder.graph
        output_node = tf_graph_builder.get_output_nodes()
        with tf.Session(graph=tf_graph) as sess:
            sess.run(tf.global_variables_initializer())  # 初始化全局变量
            results = sess.run(output_node, feed_dict=feed_dict)
            return results

    def start_test(self):
        # 从--graph-path指定路径下找到ascgen目录
        ascgraph_path_list = self._get_ascgraph_path()
        self._map_opname_to_dump_data()
        self.output_path = check_output_path_legality(self.output_path)
        real_output_path = os.path.realpath(self.output_path)
        self.npy_path = os.path.join(real_output_path, "tmp")
        for ascgraph in ascgraph_path_list:
            pyautofuse_graph = self._load_pyautofuse_graph(ascgraph)
            # 解析pyautofuse_graph，并转换为TF graph
            tf_graph_builder = convert_to_tf_graph(pyautofuse_graph)
            logger.info(f"start to run fusion op: {tf_graph_builder.graph_name}.")
            feed_dict, ge_output = self._get_ascgraph_dump_data(tf_graph_builder)
            if not feed_dict:
                continue
            golden_output = self._run_tfgraph(tf_graph_builder, feed_dict)
            if len(ge_output) != len(golden_output):
                logger.warning(f"The output length of golden and GE are not the same. "
                               "Fusio op name: {tf_graph_builder.graph_name}.")
                continue
            self._compare_output(ge_output, golden_output, tf_graph_builder)
        self._save_compare_result()
        if os.path.exists(os.path.join(self.output_path, "tmp")):
            shutil.rmtree(os.path.join(self.output_path, "tmp"))  # 清理临时文件

    def _get_ascgraph_path(self):
        ascgen_path_list = []
        ascgraph_path_list = []
        for element in os.listdir(self.graph_path):
            full_path = os.path.join(self.graph_path, element)
            if os.path.isdir(full_path) and element.startswith("ascgen_dump_pid"):
                ascgen_path_list.append(full_path)
        # filter fusedascbackend node
        for ascgen_path in ascgen_path_list:
            graph_path = get_ascbackend_ascgraph(ascgen_path)
            if len(graph_path) != 1:
                logger.warning("Pleas check your ascgen_dump path: %r. We "
                               "find %d AscGraph files." % (ascgen_path, len(graph_path)))
                continue
            ascgraph_path_list.append(os.path.join(ascgen_path, graph_path[0]))
        return ascgraph_path_list

    def _map_opname_to_dump_data(self):
        for data in os.listdir(self.input_path):
            full_path = os.path.join(self.input_path, data)
            file_name_split = data.split(".")
            # GE dump数据格式为optype.opname.id.device_id.timestamp
            if len(file_name_split) != 5:
                logger.warning("Find a unsupported GE dump file: %r." % data)
                continue
            op_type = file_name_split[0]
            op_name = file_name_split[1]
            if op_type in self.SUPPORT_OP_LIST:
                self.opname_to_dump_data_map[op_name] = full_path
            
    def _get_ascgraph_dump_data(self, tf_graph_builder):
        placeholders = tf_graph_builder.list_placeholders()
        tf_graph = tf_graph_builder.graph
        graph_name = tf_graph_builder.graph_name
        feed_dict = {}
        output_file_list = []
        input_file_list = []
        # 新建数据转换临时目录
        self.npy_path = os.path.join(self.npy_path, sanitize_filename(graph_name))
        os.makedirs(self.npy_path, 0o750, exist_ok=True)

        if graph_name not in self.opname_to_dump_data_map:
            logger.warning(f"Can not found {graph_name} in GE dump data path.") 
            return None, None
        convert_ge_dump_file_to_npy(self.opname_to_dump_data_map[graph_name], self.npy_path)
        # 合理的GE dump文件解析后格式：optype.opname.idx.device_id.timestamp.{input/output}.index.npy
        for file_name in sorted(os.listdir(self.npy_path)):
            if ".input." in file_name:
                input_file_list.append(os.path.join(self.npy_path, file_name))
            elif ".output." in file_name:
                output_file_list.append(os.path.join(self.npy_path, file_name))
            else:
                continue

        for idx, pld in enumerate(placeholders):
            pld_name = pld.name
            pld_tensor = tf_graph.get_tensor_by_name(pld_name + ":0")
            shape = pld_tensor.shape.as_list()
            dtype = pld_tensor.dtype.as_numpy_dtype
            np_data = np.load(input_file_list[idx]).astype(dtype)
            if np_data.shape != shape:
                np_data = np_data.reshape(shape)
            feed_dict[pld_tensor] = np_data
        return feed_dict, output_file_list

    def _compare_output(self, ge_dump, golden_dump, tf_graph_builder):
        for golden, my_path in zip(golden_dump, ge_dump):
            ge_data = np.load(my_path).flatten()
            golden = golden.flatten()
            self.compare_result["Opname"].append(tf_graph_builder.graph_name)
            err_msg = ""
            for metric_name in self.PRECISION_METRIC:
                probe_metric, probe_msg = NP_CMP_ALG_MAP[metric_name](golden, ge_data)
                self.compare_result[metric_name].append("NaN")
                if not probe_msg:
                    self.compare_result[metric_name][-1] = probe_metric
                    err_msg += probe_msg
            self.compare_result["Fail reason"].append(err_msg)
        
    def _save_compare_result(self):
        result_df = pd.DataFrame(self.compare_result)
        cmp_result_path = os.path.join(self.output_path, "autofuse_opcheck_result.csv")
        result_df.to_csv(cmp_result_path, index=False, na_rep='NaN')
        logger.info("autofuse opcheck result save to %r." % cmp_result_path)
