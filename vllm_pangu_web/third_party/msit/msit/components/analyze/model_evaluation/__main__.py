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
import re
import argparse

from components.utils.parser import BaseCommand
from model_evaluation.common import utils
from model_evaluation.common.enum import Framework
from model_evaluation.bean import ConvertConfig
from model_evaluation.core import Analyze
from components.utils.file_open_check import FileStat
from components.utils.log import logger
from components.utils.file_utils import check_output_dir_path


MAX_SIZE_LIMITE_NORMAL_MODEL = 32 * 1024 * 1024 * 1024  # 10G 普通模型文件


def check_model_path_legality(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal('read'):
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_type(["onnx", "prototxt", "pb"]):
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_size(MAX_SIZE_LIMITE_NORMAL_MODEL):
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
    return path_value


def check_weight_path_legality(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"weight path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal('read'):
        raise argparse.ArgumentTypeError(f"weight path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_type(["caffemodel"]):
        raise argparse.ArgumentTypeError(f"weight path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_size(MAX_SIZE_LIMITE_NORMAL_MODEL):
        raise argparse.ArgumentTypeError(f"weight path:{path_value} is illegal. Please check.")
    return path_value


def check_soc_string(value):
    if not value:
        return value
    soc_string = value
    regex = re.compile(r"[^_A-Za-z0-9\-]")
    if regex.search(soc_string):
        raise argparse.ArgumentTypeError(f"dym string \"{soc_string}\" is not a legal string")
    return soc_string


def check_output_path_legality(value):
    if not value:
        return value
    path_value = value
    check_output_dir_path(path_value)
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError("Output path is illegal, please check.") from err
    if not file_stat.is_basically_legal("write"):
        raise argparse.ArgumentTypeError("The current output path does not have right write permission, please check.")
    return path_value


def parse_input_param(model: str, framework: str, weight: str, soc: str) -> ConvertConfig:
    if framework is None:
        framework = utils.get_framework(model)
        if framework == Framework.UNKNOWN:
            raise ValueError('parse framework failed, use --framework.')
    else:
        if not framework.isdigit():
            raise ValueError('framework is illegal, use --help.')
        framework = Framework(int(framework))

    if not soc:
        soc = utils.get_soc_type()

    return ConvertConfig(framework=framework, weight=weight, soc_type=soc)


class AnalyzeCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "-gm",
            "--golden-model",
            type=check_model_path_legality,
            required=True,
            default=None,
            help="model path, support caffe, onnx, tensorflow.",
        )
        parser.add_argument(
            "--framework",
            type=str,
            choices=['0', '3', '5'],
            default=None,
            help="Framework type: 0:Caffe; 3:Tensorflow; 5:Onnx.",
        )
        parser.add_argument(
            "-w",
            "--weight",
            type=check_weight_path_legality,
            required=False,
            default='',
            help="Weight file. Required when framework is Caffe.",
        )
        parser.add_argument(
            "-soc", "--soc-version", type=check_soc_string, required=False, default='', help="The soc version."
        )
        parser.add_argument(
            "-o", "--output", type=check_output_path_legality, required=True, default='', help="Output path."
        )

    def handle(self, args):
        input_model = args.golden_model
        framework = args.framework
        weight = args.weight
        soc_version = args.soc_version
        output = args.output

        if not os.path.isfile(input_model):
            logger.error('input model is not file.')
            return

        try:
            config = parse_input_param(input_model, framework, weight, soc_version)
        except ValueError as e:
            logger.error(f'{e}')
            return

        analyzer = Analyze(input_model, output, config)
        if analyzer is None:
            logger.error('the object of \'Analyze\' create failed.')
            return

        analyzer.analyze_model()
        logger.info('analyze model finished.')


def get_cmd_instance():
    help_info = "Analyze tool to evaluate compatibility of model conversion"
    cmd_instance = AnalyzeCommand("analyze", help_info)
    return cmd_instance
