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
import unittest
import argparse
import torch
import numpy as np

from components.debug.common import logger
from components.utils.cmp_algorithm import NP_CMP_ALG_MAP, CUSTOM_ALG_MAP
from components.utils.util import load_file_to_read_common_check
from msit_opcheck.utils import NAMEDTUPLE_PRECISION_METRIC, NAMEDTUPLE_PRECISION_MODE
from msit_opcheck.conversion.shape_convert import is_transformable, format_transformation_map

FLOAT_EPSILON = np.finfo(float).eps


class OperationTest(unittest.TestCase):
    def __init__(self, methodName='opTest', case_info=None):
        super(OperationTest, self).__init__(methodName)

        self.case_info = case_info
        self.case_info['res_detail'] = []
        self.op_type = case_info['op_type'][0]
        self.op_name = case_info['op_name']
        self.op_param = case_info['op_param']
        self.base_path = case_info['base_path']
        self.data_path_dict = case_info['data_path_dict']
        self.in_tensors = []
        self.out_tensors = []
        self.bind_idx = []

        error1 = 'Error0.1‰'
        error2 = 'Error0.5‰'
        error3 = 'Error1‰'
        error4 = 'Error4‰'
        error5 = 'Error5‰'
        error6 = 'Error+/-1'

        self.precision_standard = {
            'float64': [error1, 99.99], 'uint32': [error1, 99.99], 'int64': [error1, 99.99],
            'float32': [error1, 99.99], 'int32': [error1, 99.99], 'uint64': [error1, 99.99],
            'float16': [error3, 99.9], 'uint8': [error6, 99], 'int8': [error6, 99.9],
            'bfloat16': [error4, 99.6], 'int16': [error6, 99.9], 'uint16': [error6, 99.9],
            'bool': [error1, 100]
        }

        self.erol_dict = {
            error1: 0.0001,
            error2: 0.0005,
            error3: 0.001,
            error4: 0.004,
            error5: 0.005,
            error6: 1
        }

    def validate_param(self, *param_names):
        ret = True
        for param_name in param_names:
            param = self.op_param.get(param_name, None)
            if param is None:
                ret = False
                msg = f"Cannot get golden data because opParam {param_name} is not correctly set!"
                logger.error(msg)
        return ret

    def validate_int_range(self, param_value, int_range, param_name=''):
        try:
            ivalue = int(param_value)
        except ValueError as e:
            logger_text = f"Error: The value '{param_value}' cannot be converted to an integer."
            logger.error(logger_text)
            raise RuntimeError(logger_text) from e

        if ivalue not in int_range:
            error_msg = f"[{param_name}]{param_value} is not in range {int_range}!"
            raise argparse.ArgumentTypeError(error_msg)

    def validate_path(self, path):
        if not path or not os.path.exists(path):
            raise RuntimeError(f"{path} not valid")

    def get_tensor_path(self, path, tensor_type):
        tensor_files = []
        for data_path in self.data_path_dict[tensor_type]:
            if os.path.exists(os.path.join(path, data_path)):
                cur_path = os.path.join(path, data_path)
                cur_path = load_file_to_read_common_check(cur_path)
                tensor_files.append(cur_path)
            else:
                raise RuntimeError(f"{data_path} not valid")
        return tensor_files

    def read_tensor_from_file(self, tensor_files):
        res = []
        for tensor_file in tensor_files:
            tensor_file = os.path.realpath(tensor_file)
            tensor_file = load_file_to_read_common_check(tensor_file)
            tensor = np.load(tensor_file)
            res.append(tensor)
        return res

    def force_dtype(self, tensors, precision_mode):
        float_types = (torch.float, torch.float32, torch.float16, torch.half, torch.bfloat16)
        if precision_mode == NAMEDTUPLE_PRECISION_MODE.force_fp16:
            return [t.to(torch.float16) if t.dtype in float_types else t for t in tensors]
        elif precision_mode == NAMEDTUPLE_PRECISION_MODE.force_fp32:
            return [t.to(torch.float32) if t.dtype in float_types else t for t in tensors]
        else:
            return tensors
    
    def setUp(self):
        # read input & output data
        self.validate_path(self.base_path)
        _in_tensor_files = self.get_tensor_path(self.base_path, "input")
        self.in_tensors = self.read_tensor_from_file(_in_tensor_files)
        self.in_tensors = self.force_dtype(self.in_tensors, self.case_info['precision_mode'])

        _out_tensor_files = self.get_tensor_path(self.base_path, "output")
        self.out_tensors = self.read_tensor_from_file(_out_tensor_files)
        self.out_tensors = self.force_dtype(self.out_tensors, self.case_info['precision_mode'])

    def tearDown(self):
        if self.case_info['excuted_information'] != 'PASS':
            self.case_info['excuted_information'] = 'FAILED'

    def tensor_format_transform(self, x, desc_type, order):
        x_ori_shape, x_ori_format = None, None
        x_new_format = self.op_param[desc_type][order]['layout']
        for attr in self.op_param[desc_type][order]['attr']:
            if attr['key'] == 'origin_format':
                x_ori_format = attr['value']['s']
            if attr['key'] == 'origin_shape':
                if 'i' in attr['value']['list'].keys():
                    x_ori_shape = attr['value']['list']['i']
        if x_ori_shape and is_transformable(x_new_format, x_ori_format):
            x = format_transformation_map[x_new_format][x_ori_format](x, x_new_format, x_ori_shape)
        return x

    def excute_common(self):
        logger_text = f"———————— {self.op_type} {self.op_name} test start ————————"
        logger.info(logger_text)

        for i in range(min(len(self.op_param['input_desc']), len(self.in_tensors))):
            self.in_tensors[i] = self.tensor_format_transform(self.in_tensors[i], 'input_desc', i)

        try:
            golden_out_tensors = self.golden_calc(self.in_tensors)
        except ZeroDivisionError as e:
            error_text = f"get ZeroDivisionError when calc {self.op_name} golden"
            self.case_info['fail_reason'] = "ZeroDivisionError when calc golden"
            raise RuntimeError(error_text) from e
        except IndexError as e:
            error_text = f"get IndexError when calc {self.op_name} golden: {str(e)}"
            self.case_info['fail_reason'] = "IndexError when calc golden"
            raise RuntimeError(error_text) from e
        except Exception as e:
            error_text = f"Unexpected Error when calc {self.op_name} golden: {str(e)}"
            self.case_info['fail_reason'] = "Unexpected Error when calc golden"
            raise RuntimeError(error_text) from e

        for i in range(min(len(self.op_param['output_desc']), len(self.out_tensors))):
            self.out_tensors[i] = self.tensor_format_transform(self.out_tensors[i], 'output_desc', i)
        out_tensors = self.out_tensors

        try:
            logger_text1 = f"out_tensor: {out_tensors[0].size}"
            logger_text2 = f"golden_out_tensor: {golden_out_tensors[0].size}"
            logger.debug(logger_text1)
            logger.debug(logger_text2)
        except TypeError as e:
            logger_text = "The output is abnormal. Please check! Exception: {}".format(e)
            logger.debug(logger_text)

        self.__golden_compare_all(out_tensors, golden_out_tensors)

    def execute(self):
        self.excute_common()

    def get_rel_pass_rate(self, out, golden, etol):
        out, golden = out.ravel(), golden.ravel()
        size = out.size

        rel_errors = np.where(
            np.abs(golden) > FLOAT_EPSILON,
            np.abs(out / golden - 1),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
            np.zeros_like(out, dtype=out.dtype),
        )
        rel_pass_rate = np.sum(rel_errors <= etol) / size if size != 0 else 0
        max_rel_error = np.max(rel_errors)
        return rel_pass_rate * 100, max_rel_error

    def get_abs_pass_rate(self, out, golden, etol):
        size = out.size
        abs_errors = np.where(
            np.abs(golden) > FLOAT_EPSILON,
            np.abs(out - golden),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
            np.zeros_like(out, dtype=out.dtype),
        )
        abs_pass_rate = np.sum(abs_errors <= etol) / size if size != 0 else 0
        max_abs_error = np.max(abs_errors)
        return abs_pass_rate * 100, max_abs_error

    def get_other_precisions(self, out, golden, etol):
        message = []
        precision_metric = self.case_info['precision_metric']
        default_str = 'NaN'
        abs_pass_rate, max_abs_error, cos_sim, kl = None, None, None, None

        out, golden = out.ravel().astype(np.float64), golden.ravel().astype(np.float64)
        try:
            if NAMEDTUPLE_PRECISION_METRIC.abs in precision_metric:
                abs_pass_rate, max_abs_error = self.get_abs_pass_rate(out, golden, etol)
        except Exception as e:
            logger_text = f"get_abs_pass_rate error: {e}"
            logger.error(logger_text)
        try:
            if NAMEDTUPLE_PRECISION_METRIC.cos_sim in precision_metric:
                cos_sim, cur_message = NP_CMP_ALG_MAP["cosine_similarity"](golden, out)
                if cur_message:
                    message.append('cos_sim: ' + cur_message)
        except Exception as e:
            logger_text = f"get_cosine_similarity error: {e}"
            logger.error(logger_text)
        try:
            if NAMEDTUPLE_PRECISION_METRIC.kl in precision_metric:
                kl, cur_message = NP_CMP_ALG_MAP["kl_divergence"](golden, out)
                if cur_message:
                    message.append('kl_div: ' + cur_message)
        except Exception as e:
            logger_text = f"get_kl_divergence error: {e}"
            logger.error(logger_text)

        abs_pass_rate_str = "%.16f" % float(abs_pass_rate) if abs_pass_rate is not None else default_str
        max_abs_error_str = "%.16f" % float(max_abs_error) if max_abs_error is not None else default_str
        cos_sim_str = "%.10f" % cos_sim if cos_sim is not None else default_str
        kl_div_str = "%.16f" % kl if kl is not None else default_str

        return (abs_pass_rate_str, max_abs_error_str, cos_sim_str, kl_div_str), ", ".join(message)

    def __golden_compare_all(self, out_tensors, golden_out_tensors):
        message, pass_flag = [], True

        my_data_len, golden_data_len = len(out_tensors), len(golden_out_tensors)
        if my_data_len != golden_data_len:
            pass_flag = False
            logger_text = f"Data count not equal, {my_data_len} != {golden_data_len}. Will compare only partial"
            logger.info(logger_text)

        for out_tensor, golden_out_tensor in zip(out_tensors, golden_out_tensors):
            out_dtype = str(out_tensor.dtype)
            p_s = self.precision_standard.get(out_dtype, [])
            if len(p_s) != 2:
                cur_message = f"{out_dtype} not supported!"
                self.case_info['fail_reason'] = cur_message
                raise RuntimeError(cur_message)
            if out_tensor.size != golden_out_tensor.size:
                cur_message = f"size of {out_tensor.size} not match {golden_out_tensor.size}!"
                self.case_info['fail_reason'] = cur_message
                raise RuntimeError(cur_message)

            etol = self.erol_dict.get(p_s[0], 0.001)
            err_rate = p_s[1]
            ps_standard = f"{err_rate}%(error<{etol})"

            rel_pass_rate, max_rel = self.get_rel_pass_rate(out_tensor, golden_out_tensor, etol)

            if err_rate > rel_pass_rate:
                pass_flag = False
                cur_message = f"relative pass rate: {rel_pass_rate} not met standard: {err_rate}."
                message.append(cur_message)
                logger.debug(cur_message)

            rel_pass_rate = "%.16f" % float(rel_pass_rate)
            max_rel = "%.16f" % float(max_rel)
            (abs_pass_rate, max_abs, cos_sim, kl_div), cur_message = self.get_other_precisions(
                out_tensor, golden_out_tensor, etol
            )
            if cur_message:
                message.append(cur_message)

            cur_result = {
                "precision_standard": ps_standard,
                "rel_pass_rate": rel_pass_rate,
                "max_rel": max_rel,
                "abs_pass_rate": abs_pass_rate,
                "max_abs": max_abs,
                "cos_sim": cos_sim,
                "kl_div": kl_div,
            }
            for name, compare_func in CUSTOM_ALG_MAP.items():
                cur_result[name], cur_message = compare_func(golden_out_tensor, out_tensor)
                if cur_message:
                    message.append(f"{name}: {cur_message}")
            self.case_info['res_detail'].append(cur_result)

            if pass_flag:
                self.case_info['excuted_information'] = 'PASS'

            else:
                self.case_info['excuted_information'] = 'FAILED'
            self.case_info['fail_reason'] = ", ".join(message)
