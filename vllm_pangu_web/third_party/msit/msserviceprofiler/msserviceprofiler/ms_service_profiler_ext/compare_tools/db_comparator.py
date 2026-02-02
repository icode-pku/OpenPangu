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

from .base import BaseComparator


class DBComparator(BaseComparator):
    SUPPORTED_EXTENSIONS = ['.db']

    def process(self, file_a, file_b):
        cursor = self.out_db_conn.cursor()
        cursor.execute(f"ATTACH '{file_a}' AS src1")
        cursor.execute(f"ATTACH '{file_b}' AS src2")

        self._merge_tables(cursor, 'modelExec')
        self._merge_tables(cursor, 'BatchSchedule')
        self.out_db_conn.commit()

    def _merge_tables(self, cursor, table_name):
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                batch_type TEXT,
                batch_size_a INTEGER,
                batch_size_b INTEGER
            )
        ''')

        cursor.execute(f'''
            INSERT INTO {table_name}
            SELECT
                src1.batch.batch_type,
                src1.batch.batch_size,
                src2.batch.batch_size
            FROM src1.batch
            INNER JOIN src2.batch
                ON src1.batch.name = src2.batch.name
                AND src1.batch.name = '{table_name}'
        ''')
