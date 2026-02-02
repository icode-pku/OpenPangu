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

import os
import json
import queue
import unittest

from msit_opcheck.golden_funcs import OP_DICT
from components.utils.cmp_algorithm import CUSTOM_ALG_MAP
from components.utils.file_open_check import sanitize_csv_value
from msit_opcheck.utils import NAMEDTUPLE_PRECISION_METRIC
from components.debug.common import logger


class CaseManager:
    def __init__(self, precision_metric, output_path='./'):
        self.precision_metric = precision_metric
        self.output_path = output_path
        self.cases = []

    @staticmethod
    def excute_case(case_queue, result_queue, log_level, custom_algorithms):
        runner = unittest.TextTestRunner(verbosity=2)
        testloader = unittest.TestLoader()
        
        while not case_queue.empty():
            try:
                case_info = case_queue.get_nowait()
                op = OP_DICT[case_info['op_type']]
                testnames = testloader.getTestCaseNames(op)
                for name in testnames:
                    op_cur = op(name, case_info=case_info)
                    runner.run(op_cur)
                    result_queue.put(op_cur.case_info)
            except queue.Empty as e:
                logger_text = f"The process exits because case_queue is empty. \nException: {e}"
                logger.debug(logger_text)
            except Exception as e:
                logger_text = f"An exception occurred during multiprocessing! \ncase_info: {case_info} \nException: {e}"
                logger.error(logger_text)

    def add_case(self, case_info):
        op_type = case_info['op_type']
        if len(op_type) > 1:
            # 融合算子
            return False, "Fusion op not supported"
        if op_type[0] not in OP_DICT.keys():
            #没有该op_name
            return False, "No golden function for op"
        else:
            if OP_DICT[op_type[0]]:
                self.cases.append(case_info)
                return True, ""
            else:
                #该算子用例未添加
                return False, "No golden function for op"

    def single_process(self):
        # 单进程执行测试用例
        runner = unittest.TextTestRunner(verbosity=2)
        testloader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for case_info in self.cases:
            op = OP_DICT[case_info['op_type'][0]]
            testnames = testloader.getTestCaseNames(op)
            try:
                for name in testnames:
                    op_cur = op(name, case_info=case_info)
                    suite.addTest(op_cur)
            except Exception as err:
                logger.error(f"{testnames} run failed.")
        
        runner.run(suite)
        self.write_op_result_to_csv(self.cases)

    def excute_cases(self, num_processes=1, log_level="info"):
        if num_processes == 1:
            self.single_process()
        else:
            raise RuntimeError("Only single-process opcheck is supported.")

    def write_op_result_to_csv(self, results):
        if len(results) == 0:
            return

        op_infos = []
        for op_result in results:
            op_info = {
                "op_type": op_result.get('op_type', ""),
                "op_name": op_result.get('op_name', ""),
                "op_param": json.dumps(op_result.get('op_param', "")),
                "tensor_path": op_result.get('data_path_dict', ""),
                "fail_reason": op_result.get('fail_reason', ""),
            }

            for v in op_info.values():
                sanitize_csv_value(v)
            if len(op_result['res_detail']) > 0:
                for cur_id, res_detail in enumerate(op_result['res_detail']):
                    op_infos.append(self._update_single_op_result(op_info, cur_id, res_detail))
            else:
                cur_id, res_detail = 'NaN', {}
                op_infos.append(self._update_single_op_result(op_info, cur_id, res_detail))

        import pandas as pd
        op_infos = pd.DataFrame(op_infos)
        columns = [
            "op_type", "op_name", "op_param", "tensor_path", "out_tensor_id",
            "rel_precision_rate(%)", "max_rel_error"
        ]
        if NAMEDTUPLE_PRECISION_METRIC.abs in self.precision_metric:
            columns.extend(['abs_precision_rate(%)', 'max_abs_error'])
        if NAMEDTUPLE_PRECISION_METRIC.cos_sim in self.precision_metric:
            columns.append('cosine_similarity')
        if NAMEDTUPLE_PRECISION_METRIC.kl in self.precision_metric:
            columns.append('kl_divergence')
        columns.extend(list(CUSTOM_ALG_MAP.keys()))
        columns.append("fail_reason")
        if not os.path.exists(self.output_path):
            op_infos.to_excel(self.output_path, sheet_name='opcheck_result', index=False, columns=columns)
            logger_text = f"Opcheck results saved to: {self.output_path}"
            logger.info(logger_text)
        else:
            with pd.ExcelWriter(self.output_path, engine='openpyxl', mode='a') as writer:
                op_infos.to_excel(writer, sheet_name='addition_failed_cases', index=False, columns=columns)

    def _update_single_op_result(self, op_info, cur_id, res_detail):
        default_str = 'NaN'
        op_info["out_tensor_id"] = cur_id
        op_info["rel_precision_rate(%)"] = res_detail.get('rel_pass_rate', default_str)
        op_info["max_rel_error"] = res_detail.get('max_rel', default_str)

        if NAMEDTUPLE_PRECISION_METRIC.abs in self.precision_metric:
            op_info['abs_precision_rate(%)'] = res_detail.get('abs_pass_rate', default_str)
            op_info['max_abs_error'] = res_detail.get('max_abs', default_str)
        if NAMEDTUPLE_PRECISION_METRIC.cos_sim in self.precision_metric:
            op_info['cosine_similarity'] = res_detail.get('cos_sim', default_str)
        if NAMEDTUPLE_PRECISION_METRIC.kl in self.precision_metric:
            op_info['kl_divergence'] = res_detail.get('kl_div', default_str)

        return op_info