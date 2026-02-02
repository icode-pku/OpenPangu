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
import datetime
from pathlib import Path
import shutil

from components.debug.common import logger
from components.utils.security_check import check_write_directory
from components.debug.compare.msquickcmp.common.args_check import check_input_path_legality
from msit_opcheck.case_manager import CaseManager
from msit_opcheck.graph_parser import get_all_opinfo, get_ge_graph_name, OpInfo
from msit_opcheck.utils import NAMEDTUPLE_PRECISION_MODE
from msit_opcheck.util.file_read import convert_ge_dump_file_to_npy


class OpChecker:
    def __init__(self, args):
        '''
        cases_info结构:
            'op_type': list
            'op_name': string
            'op_param': dict
            'tensor_path': string
        '''
        self.input_path = args.input
        self.output_path = args.output
        self.cases_info = {}  # 放算子对应信息

        self.ge_json_path = None
        self.origin_dump_path = None
        self.dump_data_path = None
        self.npy_path = None

        self.precision_metric = ['abs', 'cos_sim', 'kl']
        self.precision_mode = NAMEDTUPLE_PRECISION_MODE.keep_origin_dtype
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

    def check_input_path_argument(self, input_path):
        # 检查特定的子目录是否存在
        sub_directories = []
        for item in os.listdir(input_path):
            if os.path.isdir(os.path.join(input_path, item)):
                sub_directories.append(item)
        if set(sub_directories) != set(['input', 'model', 'dump_data']):
            raise ValueError("Please check the value of 'input' param, it looks like not valided.")
        model_path = os.path.join(input_path, "model")
        op_tensor_path = os.path.join(input_path, "dump_data", "npu")
        if not os.path.exists(op_tensor_path):
            raise ValueError("Path: %r not found, please check!" % op_tensor_path)

        # 检查model/ge_graph.json
        for item in os.listdir(model_path):
            json_path = os.path.join(model_path, item)
            json_path = check_input_path_legality(json_path)
            if os.path.isfile(json_path) and item.startswith("ge_proto_") and item.endswith(".json"):
                self.ge_json_path = json_path
        
        # 检查dump_data路径合法性
        op_tensor_sub_dir = os.listdir(op_tensor_path)
        if len(op_tensor_sub_dir) < 1:
            raise ValueError("No dump data in %r, please check!" % op_tensor_path)

        if not op_tensor_sub_dir[0].isdigit() or len(op_tensor_sub_dir) != 1: # 只有一个时间戳的子目录
            raise ValueError(
                "The dir name under %r is invalid, "
                "please confirm whether the file has been tampered with." % op_tensor_path
            )
        op_tensor_final_dir = os.listdir(os.path.join(op_tensor_path, op_tensor_sub_dir[0]))
        if len(op_tensor_final_dir) < 1:
            raise ValueError("No dump data in %r, please check!" % os.path.join(op_tensor_path, op_tensor_sub_dir[0]))

        if not op_tensor_final_dir[0].isdigit() or len(op_tensor_final_dir) != 1:
            raise ValueError(
                "The dir name under %r is invalid, "
                "please confirm whether the file has been tampered with." \
                % os.path.join(op_tensor_path, op_tensor_sub_dir[0])
            )
        self.origin_dump_path = os.path.join(op_tensor_path, op_tensor_sub_dir[0], op_tensor_final_dir[0])

    def init_output_file_path(self):
        check_write_directory(self.output_path)
        real_output_path = os.path.realpath(self.output_path)
        self.npy_path = os.path.join(real_output_path, "tmp")
        os.makedirs(self.npy_path, 0o750, exist_ok=True)

    def update_dump_data_path(self, new_dump_path):
        # 根据graph_name 更新 dump_data_path
        for _ in range(2):  # 往下找两层子目录，要求有且仅有一个子目录
            items = os.listdir(new_dump_path)
            sub_dirs = [item for item in items if os.path.isdir(os.path.join(new_dump_path, item))]
            
            if len(sub_dirs) > 1:
                logger.warning(
                    "Pleas check input_file path: %r, it has more than one subdirections." % new_dump_path
                )
            elif not sub_dirs:
                raise ValueError(
                    "Pleas check input_file path: %r, the files below look deleted." % new_dump_path
                )
            new_dump_path = os.path.join(new_dump_path, sub_dirs[0])
        self.dump_data_path = check_input_path_legality(new_dump_path)

    def clear_tmp_file(self, remove_dir=False):
        if remove_dir:
            shutil.rmtree(self.npy_path)
            return
        for filename in os.listdir(self.npy_path):
            file_path = os.path.join(self.npy_path, filename)
            try:
                # 如果是文件或链接，则删除
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f'Failed to delete {file_path}. Reason: {e}')
    
    def add_op_info_to_cases_info(self, op_info: OpInfo, data_info: dict):
        if not op_info.op_type:
            return
        if not isinstance(op_info.param, dict):
            return
        op_param = op_info.param
        op_type = op_info.op_type
        op_name = op_info.param.get("name")
        case_info = {
            'op_type': op_type, 'op_name': op_name, 'op_param': op_param, 'base_path': self.npy_path,
            'data_path_dict': data_info, 'precision_metric': self.precision_metric,
            'precision_mode': self.precision_mode
        }
        
        self.cases_info[op_name] = case_info

    def bind_op_info_to_case_info(self, npy_path, op_info_dict):
        file_names = os.listdir(npy_path)
        file_names.sort()
        data_info = {}
        for file_name in file_names:
            if len(file_name.split('.')) < 3:
                raise ValueError("Invalid npy file name, Please check it!.")
            op_name = file_name.split('.')[1]
            if op_name not in data_info:
                data_info[op_name] = {"input": [], "output": []}
            if file_name.split('.')[-3] == "input":
                data_info[op_name]["input"].append(file_name)
            else:
                data_info[op_name]["output"].append(file_name)

        graph_op_keys = set(op_info_dict.keys())
        mapped_keys = graph_op_keys & set(data_info.keys())
        for key in mapped_keys:
            if key in op_info_dict and key in data_info:
                op_info_dict[key].update_op_type()
                self.add_op_info_to_cases_info(op_info_dict[key], data_info[key])
            else:
                continue

    def start_test(self):
        # 0 参数校验
        self.check_input_path_argument(self.input_path)
        self.init_output_file_path()

        for graph_name in get_ge_graph_name(self.ge_json_path):
            if not graph_name:
                continue
            new_dump_path = os.path.join(self.origin_dump_path, graph_name)
            new_dump_path = check_input_path_legality(new_dump_path)
            if not os.path.exists(new_dump_path):
                continue
            self.update_dump_data_path(new_dump_path)

            # 1.转化为npy  将所有bin文件转换成npy文件，存放在{output_path}/tmp下
            convert_ge_dump_file_to_npy(self.dump_data_path, self.npy_path)
            # 
            result_csv_path = os.path.join(self.output_path, f"opcheck_result_{self.timestamp}.xlsx")
            case_manager = CaseManager(self.precision_metric, result_csv_path)

            # 2.遍历npy_path，将算子信息添加到self.cases_info
            op_info_dict = get_all_opinfo(self.ge_json_path, graph_name)
            self.bind_op_info_to_case_info(self.npy_path, op_info_dict)

            logger_text = f"Total {len(self.cases_info)} cases found under path: {self.dump_data_path}"
            logger.info(logger_text)

            # 3.将self.cases_info中的用例添加到case_manager
            result_info = 'excuted_information'
            for _, case_info in self.cases_info.items():
                if_successed_add_case, fail_message = case_manager.add_case(case_info)
                if if_successed_add_case:
                    case_info[result_info] = 'addition successed'
                else:
                    case_info[result_info] = 'addition failed'
                    case_info['fail_reason'] = fail_message

            # 4.执行测试用例并提供专家建议
            case_manager.excute_cases(1, "info")

            # 5.写入未添加成功的算子
            addition_failed_cases = []
            for v in self.cases_info.values():
                if v[result_info] == 'addition failed':
                    v['res_detail'] = []
                    addition_failed_cases.append(v)
            if len(addition_failed_cases) > 0:
                case_manager.write_op_result_to_csv(addition_failed_cases)

            self.clear_tmp_file()
        self.clear_tmp_file(True)
