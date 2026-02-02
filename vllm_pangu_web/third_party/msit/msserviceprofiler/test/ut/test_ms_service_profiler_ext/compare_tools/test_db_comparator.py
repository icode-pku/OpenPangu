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

import os
import sqlite3
import unittest
import tempfile

from msserviceprofiler.ms_service_profiler_ext.compare_tools.db_comparator import DBComparator


class TestDBComparatorIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

        self.db_a = os.path.join(self.test_dir.name, 'a.db')
        self.db_b = os.path.join(self.test_dir.name, 'b.db')
        self.output_db = os.path.join(self.test_dir.name, 'output.db')

        self.output_conn = sqlite3.connect(self.output_db)
        self._create_test_dbs()

    def test_supports_db_extension(self):
        self.assertTrue(DBComparator.supports('.db'))
        self.assertFalse(DBComparator.supports('.csv'))
        self.assertFalse(DBComparator.supports('.xlsx'))

    def test_merge_tables_with_real_dbs(self):
        comparator = DBComparator(self.output_conn, None)
        comparator.process(self.db_a, self.db_b)

        with self.output_conn:
            cursor = self.output_conn.cursor()

            cursor.execute("PRAGMA table_info(modelExec)")
            columns = [col[1] for col in cursor.fetchall()]
            self.assertEqual(columns, ['batch_type', 'batch_size_a', 'batch_size_b'])

            cursor.execute("SELECT * FROM modelExec ORDER BY batch_type")
            model_exec_data = cursor.fetchall()
            self.assertEqual(model_exec_data, [
                ('TypeA', 100, 150)
            ])

            cursor.execute("SELECT * FROM BatchSchedule ORDER BY batch_type")
            batch_schedule_data = cursor.fetchall()
            self.assertEqual(batch_schedule_data, [
                ('TypeB', 200, 250)
            ])

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cursor.fetchall()]
            self.assertCountEqual(tables, ['modelExec', 'BatchSchedule'])

    def tearDown(self):
        self.output_conn.close()
        self.test_dir.cleanup()

    def _create_test_dbs(self):
        with sqlite3.connect(self.db_a) as conn:
            conn.execute("CREATE TABLE batch (name TEXT, batch_type TEXT, batch_size INTEGER)")
            conn.executemany(
                "INSERT INTO batch VALUES (?, ?, ?)",
                [
                    ('modelExec', 'TypeA', 100),
                    ('BatchSchedule', 'TypeB', 200),
                    ('OtherTable', 'TypeC', 300)
                ]
            )
            conn.commit()

        with sqlite3.connect(self.db_b) as conn:
            conn.execute("CREATE TABLE batch (name TEXT, batch_type TEXT, batch_size INTEGER)")
            conn.executemany(
                "INSERT INTO batch VALUES (?, ?, ?)",
                [
                    ('modelExec', 'TypeA', 150),
                    ('BatchSchedule', 'TypeB', 250),
                    ('OtherTable', 'TypeC', 350)
                ]
            )
            conn.commit()
