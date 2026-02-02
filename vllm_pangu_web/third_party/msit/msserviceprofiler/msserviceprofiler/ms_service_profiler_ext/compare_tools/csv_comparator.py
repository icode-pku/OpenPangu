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
from pathlib import Path

import pandas as pd
from msserviceprofiler.msguard.security import sanitize_csv_value

from .base import BaseComparator
from ..common.utils import logger
from ..common.csv_fields import BaseCSVFields, ServiceCSVFields


class CSVComparator(BaseComparator):
    SUPPORTED_EXTENSIONS = ['.csv']

    def process(self, file_a, file_b):
        df = self.compare_csv(file_a, file_b)
        self._save_results(df, file_a)

    def compare_csv(self, file_a, file_b):
        try:
            df_a = pd.read_csv(file_a)
            df_b = pd.read_csv(file_b)
        except Exception as ex:
            logger.error(f'failed to read csv, please check {file_a}.')
            return pd.DataFrame()

        # 确保列名一致
        if set(df_a.columns) != set(df_b.columns):
            logger.error("两个 CSV 文件的列名不一致！")
            return pd.DataFrame()

        # 按 Metric 列合并两个 DataFrame
        df_merged = pd.merge(df_a, df_b, on=BaseCSVFields.METRIC, suffixes=('_a', '_b'))

        # 计算差值
        for col in df_a.columns[1:]:  # 跳过 Metric 列
            got_error = False
            try:
                # 计算绝对差值
                abs_diff = (df_merged[f'{col}_a'] - df_merged[f'{col}_b'])

                # 计算百分比相对差值（避免除以零错误）
                rel_diff = abs_diff / df_merged[f'{col}_a'].replace(0, pd.NA) * 100
                rel_diff = rel_diff.fillna(0)  # 将 NaN 替换为 0

                # 合并为字符串格式
                df_merged[f'{col}_diff'] = abs_diff.round(2).astype(str) + '|' + rel_diff.round(2).astype(str) + '%'
            except Exception as ex:
                error_msg = f'Calculate Diff Error: f{ex}'
                df_merged[f'{col}_diff'] = error_msg
                if not got_error:
                    logger.warning(error_msg)
                    got_error = True

        # 存储所有行的列表
        rows = []

        # 遍历合并后的 DataFrame
        for _, row in df_merged.iterrows():
            metric = row[BaseCSVFields.METRIC]
            metric = sanitize_csv_value(metric, replace=True)

            # 添加 a 数据、b 数据 和 diff 数据
            a_row = {BaseCSVFields.METRIC: metric, 'Data Source': 'Input Data'}
            b_row = {BaseCSVFields.METRIC: metric, 'Data Source': 'Golden Data'}
            diff_row = {BaseCSVFields.METRIC: metric, 'Data Source': 'Different'}

            for col in df_a.columns[1:]:
                # 全部 csv 字段过滤
                val_a = sanitize_csv_value(row[f'{col}_a'], replace=True)
                val_b = sanitize_csv_value(row[f'{col}_b'], replace=True)
                val_diff = sanitize_csv_value(row[f'{col}_diff'], replace=True)
                col = sanitize_csv_value(col, replace=True)

                a_row[col] = val_a
                b_row[col] = val_b
                diff_row[col] = val_diff

            rows.append(a_row)
            rows.append(b_row)
            rows.append(diff_row)

        # 将所有行转换为 DataFrame
        result = pd.DataFrame(rows)
        return result

    def _save_visualization_database(self, df, sheet_name):
        if df.shape[0] == 0:
            return
        if sheet_name == "service":
            for i in range(0, len(df), 3):
                if i + 3 <= len(df):
                    new_col_name = df.loc[i, ServiceCSVFields.METRIC]
                    df[new_col_name] = df.iloc[i:i + 3, :][ServiceCSVFields.VALUE].reset_index(drop=True)
            df.drop(columns=[ServiceCSVFields.METRIC, ServiceCSVFields.VALUE], inplace=True)
            df = df.iloc[:3, :]

        df.to_sql(name=sheet_name, con=self.out_db_conn, if_exists='replace', index=False)

    def _save_results(self, df, source_file):
        sheet_name = Path(source_file).stem.split('_')[0]
        df.to_excel(self.excel_writer, sheet_name=sheet_name, index=False)
        self._save_visualization_database(df, sheet_name)
