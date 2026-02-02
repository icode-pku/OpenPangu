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
import subprocess
from functools import reduce
from pathlib import Path

import pandas as pd
import numpy as np

from components.utils.util import filter_cmd
from components.utils.log import logger
from msit_prof.analyze.parser.ge_graph import get_all_subgraph


AUTO_FUSED_OP_TYPE = ["AscBackend", "FusedAscBackend"]
PERFORMANCE_COLUMNS = [
    "Op Name", 
    "OP Type",
    "Task Duration(us)",
    "Input Shapes",
    "Input Data Types",
    "Output Shapes",
    "Output Data Types"
]
NO_PROF_OP_TYPE_LIST = ["Reshape", "ExpandDims"]
OP_TYPE_TO_BYTE = {
    'FLOAT': 2,
    'INT64': 8,
    'INT32': 4,
    'INT8': 1,
    'BOOL': 1
}
KB_DIVISOR = 1024


class SingleOpAnalyzer:
    def __init__(self, args):
        self.fused_op_summary = args.fused
        self.origin_op_summary = args.origin
        self.output_path = args.output
        self.ge_graph_paths = args.ops_graph
        self.fused_df = None
        self.origin_df = None
        self.fused_graph_to_origin_op_mapping = None
        self.fused_name_to_type = dict()

    @staticmethod
    def calculate_gm(data_type, data_shape):
        total_gm = 0.0
        if_empty = not data_shape or not data_type
        if pd.isna(data_type) or pd.isna(data_shape) or if_empty:
            logger.debug("data_type or data_shape is missing (None or NaN).")
            return total_gm
        data_type = data_type.split(";")
        data_shape = data_shape.split(";")
        for dt, ds in zip(data_type, data_shape):
            if not isinstance(ds, str) or ds.strip(' "\'') == '':
                continue
            else:
                shape_list = list(map(int, ds.strip('"').split(',')))
                size = reduce(lambda x, y: x * y, shape_list, 1)
                total_gm += float(size * OP_TYPE_TO_BYTE[dt] / KB_DIVISOR)
        return total_gm

    @staticmethod
    def extract_attr_value(op):
        origin_op_list = None
        origin_op_type_list = None
        for attr in op["attr"]:
            if attr['key'] == "_datadump_original_op_names":
                origin_op_list = attr['value']["list"]['s']
            if attr['key'] == "_datadump_original_op_types":
                origin_op_type_list = attr['value']["list"]['s']
        return origin_op_list, origin_op_type_list

    @staticmethod
    def format_origin_ops(group: pd.DataFrame):
        deduped = group.drop_duplicates(subset='origin_op_name', keep='first')
        ops = [f"({row['origin_op_name']}, {row['origin_op_type']})" for _, row in deduped.iterrows()]
        return "; ".join(ops)

    @staticmethod
    def format_origin_duration_each_op(group: pd.DataFrame):
        op_duration = group.groupby("origin_op_name")["Task Duration(us)"].sum().reset_index()
        duration_parts = [
            f"({row['origin_op_name']}, {row['Task Duration(us)']})"
            for _, row in op_duration.iterrows()
        ]
        origin_duration_each_op = "; ".join(duration_parts)
        return origin_duration_each_op

    @staticmethod
    def format_not_found_origin_op(group: pd.DataFrame) -> str:
        """
        有些算子不会在device上计算，因此采集不到profiling数据，要进行过滤
        """
        unmatched = group[
            (group['_merge'] == 'left_only') &
            (~group['origin_op_type'].isin(NO_PROF_OP_TYPE_LIST)) & group['origin_op_name'].notna()
        ]
        deduped = unmatched.drop_duplicates(subset='origin_op_name', keep='first')
        return deduped['origin_op_name'].tolist()

    @staticmethod
    def safe_div(dividend, divisor):
        if abs(divisor) < 1e-15:
            return "NaN"
        return f"{dividend / divisor}"

    def convert_ge_graph(self):
        mapping_records = []
        for graph_path in self.ge_graph_paths:
            ops_mapping_json = os.path.join(self.output_path, Path(graph_path).stem + ".json")
            atc_cmd = [
                "atc", "--mode=5", "--om=" + graph_path, "--json=" + ops_mapping_json
            ]
            atc_cmd = filter_cmd(atc_cmd)
            res = subprocess.run(atc_cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if res.returncode == 0:
                logger.info(f"atc conversion success for graph {graph_path}.")
            else:
                raise RuntimeError("atc run failed! Make sure that you have correctly activate CANN environments.")
            mapping_records.extend(self.get_fused_graph_to_origin_op_mapping(ops_mapping_json))
        self.fused_graph_to_origin_op_mapping = pd.DataFrame(mapping_records).drop_duplicates().reset_index(drop=True)

    def get_fused_graph_to_origin_op_mapping(self, ops_mapping_json):
        records = []
        for graph in get_all_subgraph(ops_mapping_json):
            for op in graph['op']:
                if op['type'] not in AUTO_FUSED_OP_TYPE:
                    continue
                fused_op_name = op['name']
                origin_op_list, origin_op_type_list = self.extract_attr_value(op)
                if not origin_op_list or not origin_op_type_list:
                    err_msg = f"{fused_op_name} must both have '_datadump_original_op_names'" + \
                              " and '_datadump_original_op_types' attribute!"
                    raise ValueError(err_msg)
                if len(origin_op_list) != len(origin_op_type_list):
                    err_msg = f"{fused_op_name} must have the same number of " + \
                              "'_datadump_original_op_names' and '_datadump_original_op_types' attribute!"
                    raise ValueError(err_msg)
                for name, op_type in zip(origin_op_list, origin_op_type_list):
                    records.append({
                        'fused_op_name': fused_op_name,
                        'origin_op_name': name,
                        'origin_op_type': op_type
                    })
        return records

    def load_op_summary(self):
        fused_df = pd.read_csv(
            self.fused_op_summary,
            usecols=PERFORMANCE_COLUMNS
        )
        self.origin_df = pd.read_csv(
            self.origin_op_summary,
            usecols=PERFORMANCE_COLUMNS
        )
        fused_df = fused_df[fused_df["OP Type"].isin(AUTO_FUSED_OP_TYPE)]
        self.fused_df = fused_df.rename(columns={"Op Name": "fused_op_name",
                                                 "OP Type": "fused_op_type",
                                                 "Task Duration(us)": "fused_duration",
                                                 "Input Shapes": "fused_input_shapes",
                                                 "Input Data Types": "fused_input_data_types",
                                                 "Output Shapes": "fused_output_shapes",
                                                 "Output Data Types": "fused_output_data_types"
                                            })
        self.origin_df = self.origin_df.rename(columns={"Op Name": "origin_op_name"})

    def build_fusion_origin_analysis(self):
        """
        将融合后、融合前算子数据以及关联关系合并成一个DataFrame
        """
        fused_info = self.fused_df.groupby("fused_op_name").agg({
            "fused_op_type": "first",
            "fused_duration": "sum",
            "fused_input_shapes": "first",
            "fused_input_data_types": "first",
            "fused_output_shapes": "first",
            "fused_output_data_types": "first"
        }).reset_index()
        map_names = set(self.fused_graph_to_origin_op_mapping["fused_op_name"])
        fused_op_names = set(fused_info["fused_op_name"])
        dropped_fused_ops = fused_op_names - map_names
        if dropped_fused_ops:
            logger.warning(f"The following fused ops were not found in the GE dump graph: {dropped_fused_ops}")
        # 结果只保留GE dump graph中的fused ops
        analysis_df = fused_info.merge(self.fused_graph_to_origin_op_mapping, on="fused_op_name", how="left")
        analysis_df = analysis_df.merge(self.origin_df, on="origin_op_name", how="left", indicator=True)
        return analysis_df

    def calculate_origin_gm_sum(self, group: pd.DataFrame):
        deduped = group.groupby("origin_op_name").first().reset_index()
        deduped["origin_input_gm"] = deduped.apply(
            lambda row: self.calculate_gm(row["Input Data Types"], row["Input Shapes"]),
            axis=1
        )
        deduped["origin_output_gm"] = deduped.apply(
            lambda row: self.calculate_gm(row["Output Data Types"], row["Output Shapes"]),
            axis=1
        )
        origin_input_gm_sum = deduped["origin_input_gm"].sum()
        origin_output_gm_sum = deduped["origin_output_gm"].sum()
        origin_gm_each_op = [
            f'({row["origin_op_name"]}, input:{row["origin_input_gm"]}, output:{row["origin_output_gm"]})'
            for _, row in deduped.iterrows()
        ]
        return origin_input_gm_sum, origin_output_gm_sum, "; ".join(origin_gm_each_op)

    def aggregate_and_compute_group(self, group: pd.DataFrame):
        """
        Group by fused op and compute final metrics
        """
        origin_dur = group["Task Duration(us)"].sum()
        fused_dur = group["fused_duration"].iloc[0]
        fused_input_gm_sum = self.calculate_gm(group["fused_input_data_types"].iloc[0],
                                               group["fused_input_shapes"].iloc[0])
        fused_output_gm_sum = self.calculate_gm(group["fused_output_data_types"].iloc[0],
                                                   group["fused_output_shapes"].iloc[0])
        origin_input_gm_sum, origin_output_gm_sum, origin_gm_each_op = self.calculate_origin_gm_sum(group)
        not_found_op_list = self.format_not_found_origin_op(group)
        series = pd.Series({
            "Fuse OpType": group["fused_op_type"].iloc[0],
            "Origin Ops": self.format_origin_ops(group),
            "Fused Durations(us)": fused_dur,
            "Origin Durations(us)": origin_dur,
            "Time Ratio": self.safe_div(fused_dur, origin_dur),
            "Time Difference": fused_dur - origin_dur,
            "HBMs Difference": f"(input:{fused_input_gm_sum - origin_input_gm_sum}, "
                               f"output:{fused_output_gm_sum - origin_output_gm_sum})",
            "HBMs Ratio": f"(input:{self.safe_div(fused_input_gm_sum, origin_input_gm_sum)}, "
                          f"output:{self.safe_div(fused_output_gm_sum, origin_output_gm_sum)})",
            "Fused HBMs(KB)": f"(input:{fused_input_gm_sum}, output:{fused_output_gm_sum})",
            "Origin Duration(us) Each Op": self.format_origin_duration_each_op(group),
            "Origin HBMs Each Op(KB)": origin_gm_each_op,
            "Origin HBMs Total(KB)": f"(input:{origin_input_gm_sum}, output:{origin_output_gm_sum})",
            "Not Found Origin Op": "; ".join(not_found_op_list)
        })
        # 原算子一个都找不到的情况
        if (group['_merge'] == "left_only").all():
            series["Origin Durations(us)"] = np.nan
            series["Time Ratio"] = np.nan
            series["Time Difference"] = np.nan
            series["HBMs Difference"] = np.nan
            series["HBMs Ratio"] = np.nan
            series["Fused HBMs(KB)"] = np.nan
        # 原算子一部分能找到，一部分找不到的情况
        if not_found_op_list:
            series["Time Ratio"] = np.nan
            series["Time Difference"] = np.nan
            series["HBMs Difference"] = np.nan
            series["HBMs Ratio"] = np.nan
        return series

    def save_result(self, result_df: pd.DataFrame):
        analyze_result_path = os.path.join(self.output_path, "profile_analysis.csv")
        sort_df = result_df.sort_values(by="Time Difference", ascending=True, na_position='last')
        sort_df.to_csv(analyze_result_path, index=False, na_rep='NaN')
        logger.info("analyze result save to %r" % analyze_result_path)

    def analyze(self):
        # 将dump ge_build.txt转换成json文件，方便读取
        try:
            self.convert_ge_graph()
            self.load_op_summary()
            if self.fused_df.empty:
                logger.warning("No fusion ops were found.")
                return
            if self.fused_graph_to_origin_op_mapping.empty:
                logger.error("No fusion operator mapping information was found.")
                return
            analysis_df = self.build_fusion_origin_analysis()
            result_df = (
                analysis_df.groupby("fused_op_name")
                .apply(self.aggregate_and_compute_group)
                .reset_index()
                .rename(columns={'fused_op_name': 'Fuse OpName'})
            )
            self.save_result(result_df)
        except Exception as e:
            logger.error(str(e))
