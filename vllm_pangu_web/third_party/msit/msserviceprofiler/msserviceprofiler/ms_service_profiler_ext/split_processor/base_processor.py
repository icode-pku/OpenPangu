# -*- coding: utf-8 -*-
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

from abc import abstractmethod
import pandas as pd
from ms_service_profiler.exporters.utils import save_dataframe_to_csv

from ..common.constants import MAX_BATCH_NUMBER
from ..common.utils import logger
from ..common.split_utils import (
    CSV_COLUMNS, RENAMED_COLUMNS, get_statistics_data, PREFILL_NAME
)


class BaseFrameworkProcessor:
    batch_start_name = "batch_start"
    batch_end_name = "batch_end"
    http_start_name = "http_start"
    http_end_name = "http_end"
    key_name = "key"
    all_time_name = "all_time"
    name_list = [batch_start_name, http_start_name, batch_end_name]
    http_list = [http_start_name, http_end_name]
    # 记录不计算中间时间的事件(异步, 时间掩盖)或无需计算
    filter_list = [http_end_name, all_time_name]

    @staticmethod
    def _get_batch_index(full_batch, start_index, framework_df, key_pid, key_tid):
        # 从总表中获取完整的一个batch
        df_list = []
        current_index = start_index
        index_mask = framework_df.index > start_index
        for name in full_batch:
            name_mask = framework_df["name"] == name
            pid_mask = framework_df["pid"] == key_pid
            tid_mask = framework_df["tid"] == key_tid

            conditions = [
                index_mask & name_mask & pid_mask & tid_mask,
                index_mask & name_mask & pid_mask,
                index_mask & name_mask
            ]
            index = None
            for condition in conditions:
                mask = condition
                if mask.any():
                    index = framework_df[mask].index[0]
                    break
            # 找不到的点跳过
            if index is None:
                continue
            
            df_list.append(index)
            current_index = index
            # 更新index_mask
            index_mask = framework_df.index > current_index
            
        if not df_list:
            return pd.DataFrame()

        return framework_df.loc[df_list]
    
    @staticmethod
    def _calc_during_time(filter_df):
        new_rows = []
        # 遍历每个事件对
        for i in range(len(filter_df) - 1):
            current_row = filter_df.iloc[i]
            next_row = filter_df.iloc[i + 1]

            current_name = current_row["name"]
            next_name = next_row["name"]

            current_end = current_row["end_time(ms)"]
            next_start = next_row["start_time(ms)"]

            # 创建新行
            new_row = {
                "name": f"Between-{current_name}-{next_name}",
                "during_time(ms)": next_start - current_end,
                "start_time(ms)": current_end,
                "end_time(ms)": next_start,
                "pid": current_row["pid"],
                "tid": current_row["tid"],
            }
            new_rows.append(new_row)

        new_df = pd.DataFrame(new_rows)

        return new_df

    @classmethod
    @abstractmethod
    def initialize(cls, args):
        cls.args = args

    def run_split(self, framework_df, name):
        framework_df = self.preprocess_framework_df(framework_df)
        if framework_df.empty:
            return
        filter_df = self.get_filter_df(framework_df, name)
        add_all_time_df = self.get_batch_all_time(filter_df, name)
        framework_df = self.process_exporter(add_all_time_df, name)
        lower_name = name.lower()
        if self.args.log_level == "debug":
            save_dataframe_to_csv(add_all_time_df, self.args.output_path, f"{lower_name}_detail.csv")
            save_dataframe_to_csv(framework_df, self.args.output_path, f"{lower_name}_{self.args.batch_num}.csv")
        filter_name = self.http_start_name if name == PREFILL_NAME else self.batch_start_name
        framework_df = get_statistics_data(framework_df, filter_name, name)
        if not framework_df.empty:
            save_dataframe_to_csv(framework_df, self.args.output_path, f"{lower_name}.csv")

    def preprocess_framework_df(self, framework_df):
        try:
            framework_df = framework_df[framework_df["name"].isin(self.name_list)]
            framework_df = framework_df[CSV_COLUMNS]
        except KeyError as e:
            logger.warning(f"Field '{e.args[0]}' not found in datasource.")
            return pd.DataFrame()

        framework_df = framework_df.rename(columns=RENAMED_COLUMNS)
        
        return framework_df
    
    def get_filter_df(self, framework_df, name):
        """
        动态启停场景下 过滤不完整的batch
        """
        filter_name = self.http_start_name if name == PREFILL_NAME else self.batch_start_name

        valid_indices = framework_df["name"] == filter_name
        if not valid_indices.any():
            logger.warning(f"{name}: No data named {filter_name}")
            return framework_df
        
        first_index = framework_df[valid_indices].index[0]

        # 过滤 DataFrame
        return framework_df.loc[first_index:]
    
    def get_batch_all_time(self, framework_df, name):
        batch_rows = framework_df[framework_df["name"] == self.batch_start_name]
        
        if len(batch_rows) < 2:
            logger.warning(f"{name}: The length of {self.batch_start_name} is less two")
            return framework_df
        
        def create_all_time_rows(group):
            new_rows = []
            for i in range(len(group) - 1):
                current_row = group.iloc[i]
                next_row = group.iloc[i + 1]
                during_time = next_row["start_time(ms)"] - current_row["start_time(ms)"]

                all_time_row = {
                    "name": self.all_time_name,
                    "start_time(ms)": current_row["start_time(ms)"],
                    "end_time(ms)": next_row["start_time(ms)"],
                    "during_time(ms)": during_time,
                }
                new_rows.append(all_time_row)

            return pd.DataFrame(new_rows)

        all_time_dfs = batch_rows.groupby("pid").apply(create_all_time_rows).reset_index(drop=True)
        # 合并all_time行
        result_df = pd.concat([framework_df, all_time_dfs], ignore_index=True)
        result_df = result_df.sort_values(by=["start_time(ms)", "name"], ascending=[True, False]).reset_index(drop=True)

        return result_df
    
    def process_exporter(self, framework_df, name):
        # 划分组
        result_df = self._get_groups(framework_df, name)
        len_result_df = len(result_df)

        if len_result_df == 0:
            if self.args.batch_size > 0:
                size_recommend = self._get_batch_size_recommend(framework_df, name)
                logger.warning("%s: no %s with batch_size %d" % (name, self.batch_start_name, self.args.batch_size))
                if size_recommend[0] == -1:
                    logger.warning("no %s data, please check." % name)
                else:
                    logger.warning("%s: recommend batch_size from data %s" % (name, 
                                    ', '.join(map(str, size_recommend))))
            elif self.args.rid != "-1":
                logger.warning("%s: no %s with rid %r" % (name, self.batch_start_name, self.args.rid))
            return pd.DataFrame()

        merged = pd.concat(result_df, ignore_index=True)
        return merged
    
    def _get_groups(self, framework_df, name):
        result_df = []
        rid = self.args.rid
        batch_size = self.args.batch_size
        batch_num = self.args.batch_num

        def filter_by_rid(sub_group):
            filtered = sub_group[(sub_group["name"] == self.batch_start_name) &
                                 (sub_group["batch_type"] == name)]
            # 转换 rid_list 元素为字符串
            filtered.loc[:, "rid_list"] = filtered["rid_list"].apply(
                lambda x: [str(i) for i in x] if isinstance(x, list) else []
            )
            return filtered[filtered["rid_list"].apply(lambda x: rid in x)]

        def filter_by_batch_size(sub_group):
            return sub_group[(sub_group["name"] == self.batch_start_name) &
                             (sub_group["batch_type"] == name) &
                             (sub_group["batch_size"] == str(batch_size))]
        
        groups = framework_df.groupby((framework_df["name"] == self.batch_start_name).cumsum())
        result_number = 0
        for _, group in groups:
            if result_number == batch_num:
                break
            if rid != "-1":
                batch_group = filter_by_rid(group)
            elif batch_size > 0:
                batch_group = filter_by_batch_size(group)
            else:
                continue

            if batch_group.empty:
                continue

            if name == PREFILL_NAME and not self._is_valid_prefill(batch_group, framework_df):
                continue
            result = self._get_full_batch(group, framework_df)

            if not result.empty:
                result = self._get_cacl_df(result, framework_df, name)
                result_df.append(result) 
                result_number += 1
   
        return result_df

    def _is_valid_prefill(self, batch_group, framework_df):
        # batch_group不会为空且包含rid_list列
        batch_row = batch_group.iloc[0]
        cur_rid = batch_row["rid_list"][0]
        if self.args.rid != "-1":
            cur_rid = self.args.rid

        target_encode = framework_df[(framework_df["rid"] == str(cur_rid)) & 
                                     (framework_df["name"] == self.http_start_name)]
        return not target_encode.empty

    def _get_full_batch(self, group, framework_df):
        start_index = self.name_list.index(self.batch_start_name)
        end_index = self.name_list.index(self.batch_end_name)
        all_time_rows = group[group["name"] == self.all_time_name]
        if all_time_rows.empty:
            logger.debug(f"No row named {self.all_time_name} found in the group, skip this batch")
            return pd.DataFrame()
        all_time_index = all_time_rows.index[0]

        batch_start_rows = group[group["name"] == self.batch_start_name]
        if batch_start_rows.empty:
            logger.debug(f"No row named {self.batch_start_name} found in the group, skip this batch")
            return pd.DataFrame()
        batch_start_index = batch_start_rows.index[0]
        batch_rid = batch_start_rows.iloc[0]["rid"]
        
        concat_list = [batch_start_index, all_time_index]
        full_batch = self.name_list[start_index + 1: end_index + 1]

        # 找到key_row
        key_pid, key_tid = self._get_key_info(framework_df, batch_rid)
        if key_pid is None:
            logger.debug(f"no named {self.key_name} line, skip this batch")
            return pd.DataFrame()

        # 获取完整的batch
        result_index = self._get_batch_index(full_batch, batch_start_index, framework_df, key_pid, key_tid)
        if result_index.empty:
            return pd.DataFrame()
        framework_df.loc[framework_df["name"] == self.all_time_name, ["start_time(ms)", "end_time(ms)"]] = \
            framework_df.loc[framework_df["name"] == self.all_time_name, ["end_time(ms)", "start_time(ms)"]].values
        result = pd.concat([framework_df.loc[concat_list], result_index])
        result = result.sort_values(by=["start_time(ms)", "name"], ascending=[True, False]).reset_index(drop=True)
        return result
    
    def _get_key_info(self, framework_df, batch_rid):
        key_mask = ((framework_df["name"] == self.key_name) | (framework_df["name"] == "preprocess")) & \
                    (framework_df["rid"] == batch_rid)
        if not key_mask.any():
            logger.debug(f"no {self.key_name} line, skip this batch")
            return None, None
        key_row = framework_df[key_mask].iloc[0]
        key_pid, key_tid = key_row["pid"], key_row["tid"]
        return key_pid, key_tid
    
    def _get_cacl_df(self, filter_df, framework_df, name):
        empty_row = pd.DataFrame(index=[0])
        cur_rid = filter_df.iloc[0]["rid_list"][0]
        if self.args.rid != "-1":
            cur_rid = self.args.rid
        if name == PREFILL_NAME:
            http_df = framework_df[(framework_df["rid"] == str(cur_rid)) & 
                                (framework_df["name"].isin(self.http_list))]
            filter_df = pd.concat([filter_df, http_df], ignore_index=True)
            filter_df = filter_df.drop_duplicates(subset="name")
            filter_df = filter_df.sort_values(by="start_time(ms)")
        filter_df_ = filter_df[~filter_df["name"].isin(self.filter_list)]
        # 2. 当前行与下一行计算during_time
        add_df = self._calc_during_time(filter_df_)
        cur_df = pd.concat([filter_df, add_df], ignore_index=True)
        
        # 3. 与AllTime行的计算逻辑
        cur_df = self._postprocess_framework_df(cur_df, name)
        concat_df = pd.concat([empty_row, cur_df], ignore_index=True)
        return concat_df
    
    def _postprocess_framework_df(self, framework_df, name):
        if name == PREFILL_NAME:
            post_event = pd.concat([
                framework_df[framework_df["name"] == self.http_end_name],
                framework_df[framework_df["name"] == self.all_time_name]
            ])
        else:
            filter_df = framework_df[(framework_df["name"] != self.all_time_name) &
                                     (~framework_df["name"].str.startswith("Between-"))]
            if filter_df.empty:
                return pd.DataFrame()
            
            last_row = filter_df.iloc[[-1]]
            post_event = pd.concat([
                last_row,
                framework_df[framework_df["name"] == self.all_time_name]
            ])
        framework_df = framework_df.sort_values(by="start_time(ms)")
        new_rows = self._calc_during_time(post_event)

        framework_df = pd.concat([framework_df, new_rows], ignore_index=True)
        all_time_row = framework_df[framework_df["name"] == self.all_time_name]
        non_all_time_row = framework_df[framework_df["name"] != self.all_time_name]
        framework_df = pd.concat([non_all_time_row, all_time_row], ignore_index=True)
        
        return framework_df
   
    def _get_batch_size_recommend(self, framework_df, name):
        batch_df = framework_df[(framework_df["name"] == self.batch_start_name) &
                                (framework_df["batch_type"] == name)]
        if batch_df.empty:
            return [-1]
        batch_size = batch_df["batch_size"].unique()
        if len(batch_size) == 0:
            logger.warning(f"{name}: The batch_size is empty")
            return [-1]
        return batch_size