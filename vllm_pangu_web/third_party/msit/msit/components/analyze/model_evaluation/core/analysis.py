# -*- coding: utf-8 -*-
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

from typing import Dict, List

from components.utils.log import logger
from model_evaluation.common import Const
from model_evaluation.common.enum import Framework
from model_evaluation.parser import AtcErrParser, OmParser, ModelParser
from model_evaluation.bean import OpInfo, OpInnerInfo, ConvertConfig
from model_evaluation.data import Opp, OpMap
from model_evaluation.core.result import OpResult, Result
from model_evaluation.core.rule import Rule
from components.utils.check.rule import Rule as CheckRule


class Analyze:
    def __init__(self, model_path: str, out_path: str, config: ConvertConfig) -> None:
        self._model_path = model_path
        self._out_path = out_path
        self._config = config

        self._model_parser = ModelParser(model_path, out_path, config)

        self._graph = None
        if config.framework == Framework.ONNX:
            from model_evaluation.graph.onnx import OnnxGraph

            try:
                CheckRule.input_file().check(self._model_path, will_raise=True)
                self._graph: OnnxGraph = OnnxGraph.load(self._model_path)
            except RuntimeError as e:
                logger.error(f'load onnx failed, err:{e}')

        # output result
        self._result = Result()

    def analyze_model(self) -> None:
        op_infos: List[OpInfo] = self._model_parser.parse_all_ops(convert=True)
        self._init_result(op_infos)

        errcode, errinfo = self._model_parser.parse_model_to_om()
        eval_rule = Rule.get_rule_with_atc_err(errcode)
        if eval_rule == Rule.EVAL_ATC_SUCCESS:
            self._analyze_op_engine()
        elif eval_rule == Rule.EVAL_ATC_UNSUPPORT_OP_ERR:
            err_ops = AtcErrParser.parse_unsupported_op(errinfo)
            self._update_result_with_err_ops(err_ops)
            self._analyze_op_by_map_table(op_infos)
            self._check_op_constraint()
        elif eval_rule == Rule.EVAL_ATC_OTHER_ERR:
            self._analyze_op_by_map_table(op_infos)
            self._check_op_constraint()

        self._result.dump(self._out_path)

    def _init_result(self, op_infos: List[OpInfo]):
        for op_info in op_infos:
            op_result = OpResult(
                ori_op_name=op_info.op_name, ori_op_type=op_info.op_type, soc_type=self._config.soc_type
            )
            self._result.insert(op_result)

    def _update_result_with_err_op_types(self, err_op_types):
        op_infos = self._model_parser.parse_all_ops()
        for op_info in op_infos:
            if op_info.op_type not in err_op_types:
                continue
            op_result = self._result.get(op_info.op_name)
            if op_result is not None:
                op_result.is_supported = False
                op_result.set_details(Const.ERR_UNSUPPORT)
                continue
            op_result = OpResult(
                ori_op_name=op_info.op_name,
                ori_op_type=op_info.op_type,
                is_supported=False,
                details=Const.ERR_UNSUPPORT,
            )
            self._result.insert(op_result)

    def _update_result_with_err_ops(self, err_ops):
        if isinstance(err_ops, set):
            self._update_result_with_err_op_types(err_ops)
            return
        if not isinstance(err_ops, dict):
            return
        for ori_op_name, ori_op_type in err_ops.items():
            op_result = self._result.get(ori_op_name)
            if op_result is not None:
                op_result.is_supported = False
                op_result.set_details(Const.ERR_UNSUPPORT)
                continue
            op_result = OpResult(
                ori_op_name=ori_op_name, ori_op_type=ori_op_type, is_supported=False, details=Const.ERR_UNSUPPORT
            )
            self._result.insert(op_result)

    def _analyze_op_by_map_table(self, ori_op_infos: List[OpInfo]):
        try:
            op_map: OpMap = OpMap.load_op_map(self._config.framework)
        except RuntimeError as e:
            logger.error(f'load op map table failed, err:{e}.')
            return

        if self._config.framework == Framework.ONNX:
            opset_version = self._graph.opset_version()

            def map_op(op_type):
                return op_map.map_onnx_op(op_type, opset_version)

        else:

            def map_op(op_type):
                return op_map.map_op(op_type)

        try:
            opp: Opp = Opp.load_opp(self._config.soc_type, self._out_path)
        except RuntimeError as e:
            logger.warning(f'load opp data failed, err:{e}')
            return

        for ori_op_info in ori_op_infos:
            op_result = self._result.get(ori_op_info.op_name)
            inner_op = map_op(ori_op_info.op_type)
            if not isinstance(inner_op, str):
                op_result.is_supported = False
                op_result.set_details(Const.ERR_UNSUPPORT)
                continue
            op_result.op_type = inner_op
            inner_op_info = opp.query_op_info(inner_op)
            if inner_op_info.op_type == '':
                op_result.is_supported = False
                op_result.set_details(Const.ERR_UNSUPPORT)
                continue
            op_result.op_engine = inner_op_info.op_engine

    def _analyze_op_engine(self):
        om_path = self._model_parser.om_path
        if om_path is None or not os.path.isfile(om_path):
            return

        om_parser = OmParser(om_path, self._out_path)
        op_infos: List[OpInnerInfo] = om_parser.parse_all_ops(convert=True)

        def update_result(op_result: OpResult, op_info: OpInnerInfo):
            op_result.op_name = op_info.op_name
            op_result.op_type = op_info.op_type
            op_result.op_engine = op_info.op_engine

        for op_info in op_infos:
            if op_info.is_fusion:
                for ori_op in op_info.ori_ops:
                    op_result = self._result.get(ori_op)
                    if op_result is not None:
                        update_result(op_result, op_info)
            else:
                # ori_op_name is inner_op_name
                op_result = self._result.get(op_info.op_name)
                if op_result is not None:
                    update_result(op_result, op_info)

    def _check_op_constraint(self):
        if self._graph is None:
            return

        if self._config.framework == Framework.ONNX:
            from model_evaluation.core.checker import OnnxChecker
            from model_evaluation.graph.onnx import OnnxGraph

            if not isinstance(self._graph, OnnxGraph):
                return

            checker = OnnxChecker(self._graph)
            err_map: Dict[str, List[str]] = checker.check_ops()

            for ori_op, errinfos in err_map.items():
                op_result = self._result.get(ori_op)
                if op_result is None:
                    continue
                op_result.is_supported = False
                err_detail = ';'.join(errinfos)
                op_result.set_details(err_detail)
