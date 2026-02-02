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
import json
import queue
import unittest
import multiprocessing
from msit_llm.opcheck.check_case import OP_NAME_DICT
from components.utils.cmp_algorithm import CUSTOM_ALG_MAP
from msit_llm.common.utils import NAMEDTUPLE_PRECISION_METRIC
from msit_llm.common.log import logger, set_log_level


class CaseManager:
    def __init__(self, precision_metric, rerun=False, optimization_identify=False, output_path='./'):
        self.precision_metric = precision_metric
        self.rerun = rerun
        self.optimization_identify = optimization_identify
        self.output_path = output_path
        self.cases = []

    @staticmethod
    def excute_case(case_queue, result_queue, log_level, custom_algorithms):
        runner = unittest.TextTestRunner(verbosity=2)
        testloader = unittest.TestLoader()
        set_log_level(log_level)

        # Adding custom comparing algorithms
        if custom_algorithms:
            from components.utils.cmp_algorithm import register_custom_compare_algorithm

            for custom_compare_algorithm in custom_algorithms:
                register_custom_compare_algorithm(custom_compare_algorithm)
        
        while not case_queue.empty():
            try:
                case_info = case_queue.get_nowait()
                op = OP_NAME_DICT[case_info['op_name']]
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
        op_name = case_info['op_name']
        if op_name not in OP_NAME_DICT.keys():
            #没有该op_name
            return False 
        else:
            if OP_NAME_DICT[op_name]:
                self.cases.append(case_info)
                return True
            else:
                #该算子用例未添加
                return False

    def multi_process(self, num_processes=4, log_level="info", custom_algorithms=False):
        # 多进程执行测试用例
        pool = multiprocessing.get_context('spawn')
        manager = multiprocessing.Manager()
        case_queue = manager.Queue()
        result_queue = manager.Queue()
        
        # 将所有case放入队列
        for case in self.cases:
            case_queue.put(case)

        # 创建多个进程执行测试用例
        processes = []
        for _ in range(num_processes):
            process = pool.Process(target=CaseManager.excute_case, 
                                   args=(case_queue, result_queue, log_level, custom_algorithms))
            processes.append(process)
            process.start()

        # 等待所有子进程完成
        for process in processes:
            process.join()

        # 将结果写入csv文件
        results = []
        for _ in processes:
            while not result_queue.empty():
                results.append(result_queue.get())
        self.write_op_result_to_csv(results)

    def single_process(self):
        # 单进程执行测试用例
        runner = unittest.TextTestRunner(verbosity=2)
        testloader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for case_info in self.cases:
            op = OP_NAME_DICT[case_info['op_name']]
            testnames = testloader.getTestCaseNames(op)
            for name in testnames:
                op_cur = op(name, case_info=case_info)
                suite.addTest(op_cur)
        
        runner.run(suite)
        self.write_op_result_to_csv(self.cases)

    def excute_cases(self, num_processes=1, log_level="info", custom_algorithms=False):
        if num_processes == 1 or self.rerun:
            self.single_process()
        else:
            self.multi_process(num_processes, log_level, custom_algorithms)

    def write_op_result_to_csv(self, results):
        if len(results) == 0:
            return

        op_infos = []
        for op_result in results:
            op_info = {
                "op_id": op_result.get('op_id', ""),
                "op_name": op_result.get('op_name', ""),
                "op_param": json.dumps(op_result.get('op_param', "")),
                "tensor_path": op_result.get('tensor_path', ""),
                "precision_result": op_result.get('excuted_information', ""),
                "fail_reason": op_result.get('fail_reason', ""),
                "optimization_closed": op_result.get('optimization_closed', "")
            }
            
            if len(op_result['res_detail']) > 0:
                for cur_id, res_detail in enumerate(op_result['res_detail']):
                    op_infos.append(self._update_single_op_result(op_info, cur_id, res_detail))
            else:
                cur_id, res_detail = 'NaN', {}
                op_infos.append(self._update_single_op_result(op_info, cur_id, res_detail))

        import pandas as pd
        op_infos = pd.DataFrame(op_infos)
        columns = [
            "op_id", "op_name", "op_param", "tensor_path", "out_tensor_id", "precision_standard", "precision_result", 
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
        if self.optimization_identify:
            columns.append("optimization_closed")
        op_infos = op_infos.sort_values(by=['op_id', 'out_tensor_id'])
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
        op_info["precision_standard"] = res_detail.get('precision_standard', default_str)
        op_info["rel_precision_rate(%)"] = res_detail.get('rel_pass_rate', default_str)
        op_info["max_rel_error"] = res_detail.get('max_rel', default_str)

        if NAMEDTUPLE_PRECISION_METRIC.abs in self.precision_metric:
            op_info['abs_precision_rate(%)'] = res_detail.get('abs_pass_rate', default_str)
            op_info['max_abs_error'] = res_detail.get('max_abs', default_str)
        if NAMEDTUPLE_PRECISION_METRIC.cos_sim in self.precision_metric:
            op_info['cosine_similarity'] = res_detail.get('cos_sim', default_str)
        if NAMEDTUPLE_PRECISION_METRIC.kl in self.precision_metric:
            op_info['kl_divergence'] = res_detail.get('kl_div', default_str)

        for custom_name in CUSTOM_ALG_MAP:
            op_info[custom_name] = res_detail.get(custom_name, default_str)

        return op_info