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

import sys
import os
import re
import unittest
import json
import glob
import argparse
import torch
import torch_npu

from msit_llm.common.tool import read_atb_data
from msit_llm.common.log import logger
from msit_llm.common.utils import NAMEDTUPLE_PRECISION_METRIC, NAMEDTUPLE_PRECISION_MODE
from components.utils.cmp_algorithm import CMP_ALG_MAP, CUSTOM_ALG_MAP

FLOAT_EPSILON = torch.finfo(torch.float).eps


class OperationTest(unittest.TestCase):
    def __init__(self, methodName='opTest', case_info=None):
        super(OperationTest, self).__init__(methodName)

        self.case_info = case_info
        self.case_info['res_detail'] = []
        self.op_id = case_info['op_id']
        self.op_name = case_info['op_name']
        self.op_param = case_info['op_param']
        self.tensor_path = case_info['tensor_path']
        self.pid = case_info['pid']
        self.in_tensors = []
        self.out_tensors = []
        self.bind_idx = []
        self.atb_rerun = self.case_info["atb_rerun"]
        self.optimization_closed = self.case_info["optimization_closed"]

        error1 = 'Error0.1‰'
        error2 = 'Error0.5‰'
        error3 = 'Error1‰'
        error4 = 'Error4‰'
        error5 = 'Error5‰'
        error6 = 'Error+/-1'
        error7 = 'Error2^-20'
        error8 = 'Error2^-10'
        error9 = 'Error2^-8'

        self.precision_standard = {
            'torch.double': [error1, 99.99], 'torch.uint32': [error1, 99.99], 'torch.int64': [error1, 99.99],
            'torch.float32': [error7, 99.99], 'torch.int32': [error1, 99.99], 'torch.uint64': [error1, 99.99],
            'torch.float16': [error8, 99.9], 'torch.bfloat16': [error9, 99.9], 'torch.int8': [error6, 99.9],
            'torch.uint8': [error6, 99], 'torch.int16': [error6, 99.9], 'torch.uint16': [error6, 99.9],
            'torch.bool': [error1, 100]
        }

        self.erol_dict = {
            error1: 0.0001,
            error2: 0.0005,
            error3: 0.001,
            error4: 0.004,
            error5: 0.005,
            error6: 1,
            error7: 1 / (2 ** 20),
            error8: 1 / (2 ** 10),
            error9: 1 / (2 ** 8)
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
        ivalue = int(param_value)
        if ivalue not in int_range:
            error_msg = f"【{param_name}】{param_value} is not in range {int_range}!"
            raise argparse.ArgumentTypeError(error_msg)

    def validate_path(self, path):
        if not path or not os.path.exists(path):
            raise RuntimeError(f"{path} not valid")

    def get_tensor_path(self, path, tensor_type):
        if os.path.exists(os.path.join(path, 'before')) and tensor_type == 'intensor':
            path = os.path.join(path, 'before')
        else:
            path = os.path.join(path, 'after')
        _tensor_path = [x for x in os.listdir(path) if x.startswith(tensor_type)]
        _tensor_path.sort(key=lambda x: int(x.split(tensor_type)[1].split('.')[0]))
        tensor_files = [os.path.join(path, x) for x in _tensor_path]
        return tensor_files

    def read_tensor_from_file(self, tensor_files):
        res = []
        for tensor_file in tensor_files:
            tensor_file = os.path.realpath(tensor_file)
            tensor = read_atb_data(tensor_file).npu()
            res.append(tensor)
        return res

    def get_in_tensors_from_single_device(self, i, rank):
        old_did_pid = f"{rank}_{self.pid}"
        new_did_pid = str(i) + "_" + "[0-9]*"
        new_tensor_path_pattern = self.tensor_path[::-1].replace(old_did_pid[::-1], new_did_pid[::-1], 1)[::-1]
        try:
            new_tensor_path = glob.glob(new_tensor_path_pattern)[0]
        except IndexError as e:
            logger_text = f"Cannot find data on rank {i}! {self.op_name} needs tensors on all devices! Exception: {e}"
            logger.error(logger_text)
            raise RuntimeError(f"{new_tensor_path_pattern} not valid") from e
        self.validate_path(new_tensor_path)
        _in_tensor_files = self.get_tensor_path(new_tensor_path, "intensor")
        return self.read_tensor_from_file(_in_tensor_files)

    def get_rank_info(self):
        rank = self.op_param.get("rank", None)
        rank_root = self.op_param.get("rankRoot", None)
        rank_size = self.op_param.get("rankSize", None)
        return rank, rank_root, rank_size

    def get_new_in_tensors(self):
        rank, rank_root, rank_size = self.get_rank_info()
        new_in_tensors = []
        for i in range(rank_root, rank_size):
            _in_tensors = self.get_in_tensors_from_single_device(i, rank)
            new_in_tensors.extend(_in_tensors)
        return new_in_tensors

    def force_dtype(self, tensors, precision_mode):
        float_types = (torch.float, torch.float32, torch.float16, torch.half, torch.bfloat16)
        if precision_mode == NAMEDTUPLE_PRECISION_MODE.force_fp16:
            return [t.to(torch.float16) if t.dtype in float_types else t for t in tensors]
        elif precision_mode == NAMEDTUPLE_PRECISION_MODE.force_fp32:
            return [t.to(torch.float32) if t.dtype in float_types else t for t in tensors]
        else:
            return tensors

    def setUp(self):
        self.validate_path(self.tensor_path)
        _in_tensor_files = self.get_tensor_path(self.tensor_path, "intensor")
        self.in_tensors = self.read_tensor_from_file(_in_tensor_files)
        self.in_tensors = self.force_dtype(self.in_tensors, self.case_info['precision_mode'])
        if not (self.atb_rerun and 'after' not in os.listdir(self.tensor_path)):
            _out_tensor_files = self.get_tensor_path(self.tensor_path, "outtensor")
            self.out_tensors = self.read_tensor_from_file(_out_tensor_files)
            self.out_tensors = self.force_dtype(self.out_tensors, self.case_info['precision_mode'])

    def tearDown(self):
        if self.case_info['excuted_information'] != 'PASS':
            self.case_info['excuted_information'] = 'FAILED'

    def get_torch_atb_params(self, op_name):
        import copy
        params = copy.deepcopy(self.op_param)
        if op_name == "SelfAttentionOperation":
            from msit_llm.opcheck.check_case.self_attention import CalcType, KvCacheCfg, MaskType, KernelType, ClampType
            if "calcType" in params:
                params["calcType"] = CalcType(self.op_param['calcType']).name
            if "kvcacheCfg" in params:
                params["kvcacheCfg"] = KvCacheCfg(self.op_param['kvcacheCfg']).name
            if "maskType" in params:
                params["maskType"] = MaskType(self.op_param['maskType']).name
            if "kernelType" in params:
                params["kernelType"] = KernelType(self.op_param['kernelType']).name
            if "clampType" in params:
                params["clampType"] = ClampType(self.op_param['clampType']).name
        else:
            from msit_llm.opcheck.check_case.paged_attention import CompressType, MaskType, QuantType, CalcType
            if "compressType" in params:
                params["compressType"] = CompressType(self.op_param['compressType']).name
            if "maskType" in params:
                params["maskType"] = MaskType(self.op_param['maskType']).name
            if "quantType" in params:
                params["quantType"] = QuantType(self.op_param['quantType']).name
            if "calcType" in params:
                params["calcType"] = CalcType(self.op_param['calcType']).name
        params = json.dumps(params)
        return params

    def run_op_torch_atb(self, op_name):
        atb_speed_path = os.getenv("ATB_SPEED_HOME_PATH", None)
        if not atb_speed_path:
            raise RuntimeError("ATB_SPEED_HOME_PATH environment variable not valid. Try install mindie")
        sys.path.append(os.path.join(atb_speed_path, "lib"))
        try:
            from _libatb_torch import _GraphOperation as GraphOp, _BaseOperation as BaseOp
        except Exception as e:
            raise RuntimeError("import _libatb_torch failed. Try install compatible mindie") from e 

        params = self.get_torch_atb_params(op_name)
        graph_op = GraphOp('rerun_op')
        op = BaseOp(op_type=op_name.replace("Operation", ""), op_param=params, op_name=op_name)
        input_name = ['in' + str(i) for i in range(len(self.in_tensors))]
        output_name = ['out' + str(i) for i in range(len(self.out_tensors))]
        graph_op.add_input_output(input=input_name, output=output_name)
        graph_op.add_operation(op, input_name, output_name)
        graph_op.build()

        inputs, outputs, bind = {}, {}, {}
        for idx, intensor in zip(input_name, self.in_tensors):
            inputs[idx] = intensor
        for idx, outtensor in zip(output_name, self.out_tensors):
            outputs[idx] = outtensor
        for idx in self.bind_idx:
            bind_name = "in" + str(idx)
            bind[bind_name] = self.in_tensors[idx].cpu()
        _ = graph_op.forward(inputs, outputs, bind)
        return outputs.values()

    def rerun_op(self, execute_type):
        if self.op_name == "SelfAttentionOperation" or self.op_name == "PagedAttentionOperation":
            out_tensors = self.run_op_torch_atb(self.op_name)
        else:
            operation = torch.classes.OperationTorch.OperationTorch(self.op_name)
            if isinstance(self.op_param, dict):
                operation.set_param(json.dumps(self.op_param))
            elif isinstance(self.op_param, str):
                operation.set_param(self.op_param)
            if execute_type == "inplace":
                operation.execute(self.in_tensors)
                out_tensors = []
                for index in self.case_info['inplace_idx']:
                    out_tensors.append(self.in_tensors[index])
            else:
                out_tensors = operation.execute(self.in_tensors)
        return [tensor.cpu() for tensor in out_tensors]

    def excute_common(self, execute_type):
        logger_text = f"———————— {self.op_id} {self.op_name} test start ————————"
        logger.info(logger_text)

        try:
            golden_out_tensors = self.golden_calc(self.in_tensors)
        except ZeroDivisionError as e:
            error_text = f"get ZeroDivisionError when calc {self.op_name} golden"
            raise RuntimeError(error_text) from e
        except IndexError as e:
            error_text = f"get IndexError when calc {self.op_name} golden: {str(e)}"
            raise RuntimeError(error_text) from e

        if self.atb_rerun:
            if self.op_name in ("AllGatherOperation", "AllReduceOperation", "LinearParallelOperation"):
                logger_text = f"{self.op_name} needs data on all ranks and atb-rerun is unsupported. " \
                              "The dump data will be used for comparison."
                logger.warning(logger_text)
                out_tensors = self.out_tensors
            else:
                out_tensors = self.rerun_op(execute_type)
        else:
            out_tensors = self.out_tensors

        try:
            logger_text1 = f"out_tensor: {out_tensors[0].size()}"
            logger_text2 = f"golden_out_tensor: {golden_out_tensors[0].size()}"
            logger.debug(logger_text1)
            logger.debug(logger_text2)
        except TypeError as e:
            logger_text = "The output is abnormal. Please check! Exception: {}".format(e)
            logger.debug(logger_text)

        self.__golden_compare_all(out_tensors, golden_out_tensors)

    def execute(self):
        self.excute_common("common")

    def execute_inplace(self):
        self.excute_common("inplace")

    def get_rel_pass_rate(self, out, golden, etol):
        out, golden = out.reshape(-1).cpu(), golden.reshape(-1).cpu()
        size = out.shape[0]
        rel_errors = torch.where(
            torch.abs(golden) > FLOAT_EPSILON,
            torch.abs(out / golden - 1),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
            torch.tensor(0, dtype=out.dtype),
        )
        rel_pass_rate = torch.sum(rel_errors <= etol) / size if size != 0 else 0
        max_rel_error = torch.max(rel_errors)
        return rel_pass_rate.item() * 100, max_rel_error.item()

    def get_abs_pass_rate(self, out, golden, etol):
        size = out.shape[0]
        abs_errors = torch.where(
            torch.abs(golden) > FLOAT_EPSILON,
            torch.abs(out - golden),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
            torch.tensor(0, dtype=out.dtype),
        )
        abs_pass_rate = torch.sum(abs_errors <= etol) / size if size != 0 else 0
        max_abs_error = torch.max(abs_errors)
        return abs_pass_rate.item() * 100, max_abs_error.item()

    def get_other_precisions(self, out, golden, etol):
        message = []
        precision_metric = self.case_info['precision_metric']
        default_str = 'NaN'
        abs_pass_rate, max_abs_error, cos_sim, kl = None, None, None, None

        out, golden = out.reshape(-1).cpu().double(), golden.reshape(-1).cpu().double()
        try:
            if NAMEDTUPLE_PRECISION_METRIC.abs in precision_metric:
                abs_pass_rate, max_abs_error = self.get_abs_pass_rate(out, golden, etol)
        except Exception as e:
            logger_text = f"get_abs_pass_rate error: {e}"
            logger.error(logger_text)
        try:
            if NAMEDTUPLE_PRECISION_METRIC.cos_sim in precision_metric:
                cos_sim, cur_message = CMP_ALG_MAP["cosine_similarity"](golden, out)
                if cur_message:
                    message.append('cos_sim: ' + cur_message)
        except Exception as e:
            logger_text = f"get_cosine_similarity error: {e}"
            logger.error(logger_text)
        try:
            if NAMEDTUPLE_PRECISION_METRIC.kl in precision_metric:
                kl, cur_message = CMP_ALG_MAP["kl_divergence"](golden, out)
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

    def get_npu_device(self):
        npu_device = os.environ.get("NPU_DEVICE")
        if npu_device is None:
            npu_device = "npu:0"
        else:
            npu_device = f"npu:{npu_device}"
        return npu_device

    def get_soc_version(self):
        device_name = torch.npu.get_device_name()
        if re.search("Ascend910B", device_name, re.I):
            soc_version = 'Ascend910B'
        elif re.search("Ascend310P", device_name, re.I):
            soc_version = 'Ascend310P'
        else:
            raise RuntimeError(f"{device_name} is not supported")
        device_count = torch.npu.device_count()
        current_device = torch.npu.current_device()
        logger_text = "Device Properties: device_name: {}, soc_version: {}, device_count: {}, current_device: {}" \
            .format(device_name, soc_version, device_count, current_device)
        logger.debug(logger_text)
        return soc_version

    def convert_data_format(self, data):
        dim0, dim1 = data.shape[0], data.shape[1]
        remainder = dim1 % 32 if data.dtype == torch.int8 else dim1 % 16
        if remainder != 0:
            raise RuntimeError(f'The tensor with shape {data.shape} is invalid to convert format')
        if data.dtype == torch.int8:
            data = data.reshape([1, dim1 // 32, dim0, 32]).permute(0, 2, 1, 3).reshape([dim0, dim1])
        else:
            data = data.reshape([1, dim1 // 16, dim0, 16]).permute(0, 2, 1, 3).reshape([dim0, dim1])
        return data

    def nz_2_nd(self, data):
        origin_shape = data.shape
        dims = list(range(len(origin_shape)))
        last_dims = dims[-4:]

        perm = dims[:-4] + [last_dims[1]] + [last_dims[2]] + [last_dims[0]] + [last_dims[3]]
        data = data.permute(perm)
        nd_shape = data.shape[:-4] + (data.shape[-4] * data.shape[-3], data.shape[-2], data.shape[-1])
        data = data.reshape(nd_shape)
        return data

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

            etol = self.erol_dict.get(p_s[0], 0.001)
            err_rate = p_s[1]
            ps_standard = f"{err_rate}%(error<{etol})"

            rel_pass_rate, max_rel = self.get_rel_pass_rate(out_tensor, golden_out_tensor, etol)

            if err_rate >= rel_pass_rate:
                pass_flag = False
                cur_message = f"relative pass rate: {rel_pass_rate} not met standart: {err_rate}."
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
