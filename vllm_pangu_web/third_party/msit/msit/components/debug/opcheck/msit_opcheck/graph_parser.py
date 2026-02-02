# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

import json

from components.utils.file_open_check import ms_open

MAX_GE_GRAPH_SIZE = 209715200   # 200 * 1024 * 1024, 200MB


class InputOutputDesc(object):
    def __init__(self, **kwargs):
        """
        kwargs like that:
            "attr": [],
            "device_type": "NPU",
            "dtype": "DT_BOOL",
            "layout": "ND",
            "real_dim_cnt": 1,
            "shape": {
                "dim": [
                    1
                ]
            },
        """
        self.attr = kwargs.get("attr")
        self.input_output_param = kwargs


class OpInfo(object):
    def __init__(self, op_info_dict):
        """
        op_info_dict like that:
            "attr": [],
            "dst_index": [],
            "dst_name": "output",
            "has_out_attr": true,
            "id": 1,
            "input": [],
            "input_desc": [],
            "input_i": [],
            "input_name": [],
            "is_input_const": [],
            "name": "output",
            "output_desc": [],
            "output_i": [],
            "src_index": [],
            "src_name": [],
            "type": "add",
        """
        self.param = op_info_dict
        self.sub_graph_attr = None
        self.sub_graph_input = None
        self.graph_attr = None
        self.op_type = None
        self.input_desc_list = []
        self.output_desc_list = []
        self.init_input_output_desc()
        
    def init_input_output_desc(self):
        if "input_desc" in self.param:
            self.input_desc_list = [InputOutputDesc(**i) for i in self.param.get("input_desc")]
        if "output_desc" in self.param:
            self.output_desc_list = [InputOutputDesc(**i) for i in self.param.get("output_desc")]

    def update_graph_info(self, graph_info_dict, global_attr):
        self.sub_graph_attr = graph_info_dict.get("attr", [])
        self.sub_graph_input = graph_info_dict.get("input", [])
        self.graph_attr = global_attr

    def update_op_type(self):
        # 更新融合算子标记
        result_op_type = [self.param.get("type", None)]
        for attr in self.param['attr']:
            if attr['key'] == '_datadump_original_op_types':
                result_op_type = attr['value']['list']['s']
        self.op_type = result_op_type


def get_single_op_info_from_op_list(op_list, graph_info_dict, global_attr):
    op_info_dict = {}
    for op in op_list:
        op_name = op.get("name", None)
        if not op_name:
            continue
        op_name = op_name.replace('/', '_')
        op_info_dict[op_name] = OpInfo(op)
        op_info_dict[op_name].update_graph_info(graph_info_dict, global_attr)
            
    return op_info_dict


def get_ge_graph_name(json_path):
    # 解析ge_json 得到每一个子图的信息
    with ms_open(json_path, max_size=MAX_GE_GRAPH_SIZE) as f:
        ge_json_file = json.load(f)
    graph_list = ge_json_file.get("graph")
    for sub_graph in graph_list:
        ge_dump_file_name = sub_graph.get("name", None)
        yield ge_dump_file_name


def get_all_opinfo(json_path, graph_name):
    with ms_open(json_path, max_size=MAX_GE_GRAPH_SIZE) as f:
        ge_file = json.load(f)
    graph = ge_file.get("graph")
    global_attr = ge_file.get("attr")

    for sub_graph in graph:
        if sub_graph['name'] != graph_name:
            continue
        op_list = sub_graph.get("op")
        op_info_dict = get_single_op_info_from_op_list(op_list, sub_graph, global_attr)
        break
    
    return op_info_dict

    