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
import re
import json
import time
import datetime
from collections import namedtuple
import torch
import torch_npu

from components.utils.check.rule import Rule
from components.utils.file_open_check import ms_open, FileStat
from msit_llm.common.log import logger
from msit_llm.common.constant import GLOBAL_HISTORY_AIT_DUMP_PATH_LIST, RAW_INPUT_PATH
from msit_llm.common.utils import load_file_to_read_common_check, NAMEDTUPLE_PRECISION_MODE
from components.utils.constants import TENSOR_MAX_SIZE


class OpChecker:
    def __init__(self):
        '''
        cases_info结构：
            'op_id': string
            'op_name': string
            'op_param': dict
            'tensor_path': string
        '''
        self.cases_info = {}
        self.special_cases = ('KvCacheOperation', 'ReshapeAndCacheOperation', 'SelfAttentionOperation')
        self.base_path = ''
        self.pid = None
        self.input = ''
        self.output = ''
        self.output_path = ''
        self.operation_ids = ''
        self.check_ids_string = []
        self.operation_name = None
        self.check_patterns = []
        self.precision_metric = []
        self.precision_mode = NAMEDTUPLE_PRECISION_MODE.keep_origin_dtype
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.atb_rerun = False
        self.optimization_identify = False
        self.jobs = 1
        self.log_level = "info"
        self.custom_algorithms = False

    @staticmethod
    def third_party_init():
        import msit_llm
        import ctypes

        # Loading libopchecker.so with ctypes
        lib_opchecker_path = os.environ.get("AIT_OPCHECK_LIB_PATH", "")
        if not lib_opchecker_path:
            lib_path_dir = os.path.dirname(os.path.abspath(msit_llm.__file__))
            lib_opchecker_path = os.path.join(lib_path_dir, "opcheck", "libopchecker.so")

        logger_text = f"lib_opchecker_path is {lib_opchecker_path}"
        logger.info(logger_text)

        # check the path of libopchecker.so before opening it
        check_res = Rule.input_file().check(lib_opchecker_path)
        if not check_res:
            logger.error(
                "%r loading failed due to %s, check if msit_llm install correctly", lib_opchecker_path, check_res
            )
            return False

        # Loading libopchecker.so
        try:
            ctypes.cdll.LoadLibrary(lib_opchecker_path).RegisterAll()
        except Exception:
            logger_text = f"{lib_opchecker_path} loading failed, check if msit_llm installed correctly"
            logger.error(logger_text)
            return False

        # Loading libatb_speed_torch.so with torch
        atb_speed_path = os.getenv('ATB_SPEED_HOME_PATH', "")
        if not atb_speed_path:
            logger.error("ATB_SPEED_HOME_PATH is empty, check if mindie_atb_models configured correctly")
            return False

        libatb_speed_torch_path = os.path.join(atb_speed_path, 'lib', 'libatb_speed_torch.so')
        logger_text = f"libatb_speed_torch_path is {libatb_speed_torch_path}"
        logger.info(logger_text)
        if not os.path.exists(libatb_speed_torch_path):
            logger_text = f"{libatb_speed_torch_path} not exists, check if mindie_atb_models configured correctly"
            logger.error(logger_text)
            return False

        try:
            torch.classes.load_library(libatb_speed_torch_path)
        except Exception:
            logger_text = f"{libatb_speed_torch_path} loading failed, check if mindie_atb_models configured correctly"
            logger.error(logger_text)
            return False

        return True

    def get_base_path(self, cur_path):
        dirseg = cur_path.split(os.path.sep)
        if len(dirseg) >= 4 and dirseg[-3] == 'tensors' and \
            any([dirseg[-4].startswith(x) for x in GLOBAL_HISTORY_AIT_DUMP_PATH_LIST]):
            try:
                pid = dirseg[-2].split("_")[1]
            except (IndexError, AttributeError, TypeError, ValueError):
                pid = None
        elif cur_path == os.path.dirname(cur_path):
            cur_path, pid = None, None
        else:
            cur_path, pid = self.get_base_path(os.path.dirname(cur_path))
        return cur_path, pid

    def check_input_legality(self, input_path):
        ret = False
        base_path = None
        pid = None

        input_path = os.path.realpath(input_path)
        if not os.path.exists(input_path):
            logger_text = f"Input path not found: {input_path}"
            logger.error(logger_text)
            return input_path, base_path, pid, ret

        base_path, pid = self.get_base_path(input_path)

        if base_path is None:
            logger_text = f"Input path is not in msit_dump tensors directory: {input_path}"
            logger.error(logger_text)
            return input_path, base_path, pid, ret

        os.environ[RAW_INPUT_PATH] = os.path.dirname(os.path.dirname(os.path.abspath(base_path)))
        ret = True
        return input_path, base_path, pid, ret

    def args_init(self, args):
        execution_flag = True

        self.input, self.base_path, self.pid, ret = self.check_input_legality(args.input)
        if not ret:
            execution_flag = False

        self.output = os.path.realpath(args.output)
        if not os.path.exists(self.output):
            logger_text = f"Output path not found: {self.output}"
            logger.error(logger_text)
            execution_flag = False

        self.output_path = os.path.join(self.output, f"opcheck_result_{self.timestamp}.xlsx")
        self.operation_ids = args.operation_ids
        if self.operation_ids != '':
            try:
                self.check_ids_string = [x.lower().strip() for x in self.operation_ids.split(',')]
            except ValueError as e:
                logger_text = "Failed to parse operation_ids. Error: {}".format(e)
                logger.error(logger_text)
                execution_flag = False
        self.operation_name = args.operation_name
        if self.operation_name is not None:
            try:
                self.check_patterns = [x.lower().strip() for x in self.operation_name.split(',')]
            except ValueError as e:
                logger_text = "Failed to parse operation_name. Error: {}".format(e)
                logger.error(logger_text)
                execution_flag = False
        self.precision_metric = args.precision_metric
        self.precision_mode = args.precision_mode
        self.jobs = args.jobs
        self.log_level = args.log_level
        self.custom_algorithms = args.custom_algorithms

        # 指定需要使用的npu设备
        try:
            torch.npu.set_device(torch.device(f"npu:{args.device_id}"))
        except RuntimeError as e:
            logger_text = "Failed to set the device. Device_id: {}. Failed Reason: {}".format(args.device_id, e)
            logger.error(logger_text)
            execution_flag = False

        self.atb_rerun = args.atb_rerun
        self.optimization_identify = args.optimization_identify
        if self.atb_rerun:
            execution_flag_res = OpChecker.third_party_init()
            if not execution_flag_res:
                execution_flag = False
            else:
                logger_text = "Rerunning operations in atb to calculate outputs..."
                logger.info(logger_text)
        else:
            if _is_atb_only_saved_before(self.input):
                logger_text = "Only the rerun mode allows checking dumped data before executing operators."
                logger.error(logger_text)
                execution_flag = False
            else:
                logger_text = "Comparing outputs in dump data without rerunning operations in atb..."
                logger.info(logger_text)
        return execution_flag

    def start_test(self, args):
        start_time = time.time()

        # 0.初始化
        execution_flag_res = self.args_init(args)
        if not execution_flag_res:
            return

        from msit_llm.opcheck.case_manager import CaseManager
        case_manager = CaseManager(self.precision_metric, self.atb_rerun, self.optimization_identify, self.output_path)

        # 1.遍历tensor_path，将算子信息添加到self.cases_info
        self.walk_tensor_path(self.input)
        logger_text = f"Total {len(self.cases_info)} cases found under path: {self.input}"
        logger.info(logger_text)

        # 2.将self.cases_info中的用例添加到case_manager
        result_info = 'excuted_information'
        for _, case_info in self.cases_info.items():
            if_successed_add_case = case_manager.add_case(case_info)
            if if_successed_add_case:
                case_info[result_info] = 'addition successed'
            else:
                case_info[result_info] = 'addition failed'

        # 3.执行测试用例并提供专家建议
        case_manager.excute_cases(self.jobs, self.log_level, self.custom_algorithms)

        # 4.写入未添加成功的算子
        addition_failed_cases = []
        for v in self.cases_info.values():
            if v[result_info] == 'addition failed':
                v['res_detail'] = []
                addition_failed_cases.append(v)
        if len(addition_failed_cases) > 0:
            case_manager.write_op_result_to_csv(addition_failed_cases)

        # 5.计算总执行时间
        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        logger_text = f"Total cases: {len(self.cases_info)}; Total time: {total_time} seconds"
        logger.info(logger_text)

    def parse_op_id_name(self, dirpath):
        basename = os.path.basename(dirpath)
        try:
            op_name = basename.split('_')[-1]
        except IndexError:
            logger_text = f"{dirpath} is not a valid tensor dir, please check"
            logger.debug(logger_text)
            op_name = None

        rel_path = os.path.relpath(dirpath, self.base_path)
        try:
            op_id = '_'.join([x.split('_')[0] for x in rel_path.split(os.path.sep)])
        except IndexError:
            logger_text = f"{dirpath} is not a valid tensor dir, please check"
            logger.debug(logger_text)
            op_id = None

        return op_id, op_name

    def check_id_range(self, op_id):
        if op_id is None:
            return False
        if self.operation_ids == '':
            return True

        for p in self.check_ids_string:
            ret = re.match("^" + p + "(_[0-9]+){0,20}$", op_id)
            if ret:
                return True
        return False

    def check_name(self, op_name):
        if op_name is None:
            return False
        if self.operation_name is None:
            return True

        for p in self.check_patterns:
            if p in op_name.lower():
                return True
        return False

    def is_exec_node(self, case_info):
        if self.operation_ids == '' and self.operation_name is None:
            return True

        flag1 = self.check_id_range(case_info.get("op_id", None))
        flag2 = self.check_name(case_info.get("op_name", None))
        return flag1 and flag2

    def traverse_optimization(self, case_info, op_name, op_id):
        import copy
        optimization = {}
        if op_name == 'SelfAttentionOperation':
            from msit_llm.opcheck.check_case.self_attention import MaskType, KernelType, ClampType
            optimization = {
                "maskType": MaskType.MASK_TYPE_UNDEFINED.value,
                "batchRunStatusEnable": False,
                "isTriuMask": 0,
                "kernelType": KernelType.KERNELTYPE_DEFAULT.value,
                "clampType": ClampType.CLAMP_TYPE_UNDEFINED.value
            }
        elif op_name == 'PagedAttentionOperation':
            from msit_llm.opcheck.check_case.paged_attention import CompressType, MaskType, QuantType, CalcType
            optimization = {
                "maskType": MaskType.UNDEFINED.value,
                "batchRunStatusEnable": False,
                "quantType": QuantType.TYPE_QUANT_UNDEFINED.value,
                "hasQuantOffset": False,
                "compressType": CompressType.COMPRESS_TYPE_UNDEFINED.value,
                "calcType": CalcType.CALC_TYPE_UNDEFINED.value
            }

        op_param = case_info.get("op_param", None)
        idx = 0
        for key, value in optimization.items():
            if op_param.get(key, value) != value:
                idx += 1
                new_op_id = op_id + '_' + str(idx)
                new_case_info = copy.deepcopy(case_info)
                new_case_info['op_id'] = new_op_id
                new_case_info['op_param'][key] = value
                new_case_info['optimization_closed'] = key
                self.cases_info[new_op_id] = new_case_info

    def add_case_to_cases(self, case_info):
        op_name = case_info.get("op_name", None)
        op_id = case_info.get("op_id", None)
        if op_name == 'KvCacheOperation':
            case_info['inplace_idx'] = [2]
            self.cases_info[op_id] = case_info
        elif op_name == 'ReshapeAndCacheOperation':
            case_info['inplace_idx'] = [2, 3]
            self.cases_info[op_id] = case_info
        elif op_name in ['SelfAttentionOperation', 'PagedAttentionOperation']:
            self.cases_info[op_id] = case_info
            if self.optimization_identify:
                self.traverse_optimization(case_info, op_name, op_id)
        else:
            self.cases_info[op_id] = case_info

    def add_op_info_to_cases_info(self, dirpath):
        tensor_path = dirpath

        op_param = {}
        
        json_path = os.path.join(dirpath, 'op_param.json')
        if FileStat(json_path).is_exists:
            try:  
                json_path = load_file_to_read_common_check(json_path)  
                with ms_open(json_path, 'r', max_size=TENSOR_MAX_SIZE) as f:
                    op_param = json.load(f)
            except Exception as e:
                logger_text = f"Cannot loads json file to json! Json file: {json_path} \n Exception: {e}"
                logger.debug(logger_text)

        op_id, op_name = self.parse_op_id_name(dirpath)
        if op_id is None or op_name is None:
            return

        case_info = {
            'op_id': op_id, 'op_name': op_name, 'op_param': op_param, 'tensor_path': tensor_path,
            'precision_metric': self.precision_metric, 'atb_rerun': self.atb_rerun, 'pid': self.pid,
            'precision_mode': self.precision_mode, 'optimization_closed': ''
        }

        ret = self.is_exec_node(case_info)
        if ret:
            self.add_case_to_cases(case_info)
        return

    def walk_tensor_path(self, cur_path):
        from msit_llm.opcheck.check_case import OP_NAME_DICT
        files_and_dirs = os.listdir(cur_path)
        dirnames = [item for item in files_and_dirs if os.path.isdir(os.path.join(cur_path, item))]

        if any(dirname in ['after', 'before'] for dirname in dirnames):
            op_name = os.path.basename(cur_path).split('_')[-1]
            if op_name in OP_NAME_DICT:
                self.add_op_info_to_cases_info(cur_path)
        # 遍历下一级文件夹
        for dirname in dirnames:
            if dirname not in ['after', 'before']:
                self.walk_tensor_path(os.path.join(cur_path, dirname))


def _is_atb_only_saved_before(input_path):
    if not os.listdir(input_path):
        logger_text = "Input path does not contain operator folders."
        logger.error(logger_text)
        return False

    filename = os.listdir(input_path)[0]
    filepath = os.path.join(input_path, filename)

    if os.path.isdir(filepath):
        subfiles = os.listdir(filepath)
        return 'after' not in subfiles and 'before' in subfiles
    else:
        return False
