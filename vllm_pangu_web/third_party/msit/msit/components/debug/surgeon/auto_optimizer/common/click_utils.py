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
import stat
import pathlib
from functools import partial
from multiprocessing import Pool
from typing import List, Optional, Union

import re
import click

from auto_optimizer import KnowledgeFactory
from auto_optimizer.graph_optimizer.optimizer import (
    GraphOptimizer,
    InferTestConfig,
    BigKernelConfig,
    ARGS_REQUIRED_KNOWLEDGES,
)
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from components.debug.common import logger
from components.utils.security_check import ms_makedirs


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

STR_UNSAFE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")
PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")

READ_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH
WRITE_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH


def safe_string(value):
    if not value:
        return value
    if re.search(STR_UNSAFE_LIST_REGEX, value):
        raise ValueError("String parameter contains invalid characters.")
    return value


def check_input_path(input_path):
    if not os.access(input_path, os.F_OK):
        logger.error("Input path %r is not exist.", input_path)
        return False

    if not os.access(input_path, os.R_OK):
        logger.error("Input path %r is not readable.", input_path)
        return False

    return True


def check_output_model_path(output_model):
    if os.path.isdir(output_model):
        logger.error("Output path %r is a directory.", output_model)
        return False

    model_dir = os.path.dirname(os.path.abspath(output_model))
    if not os.path.exists(model_dir):
        logger.error("Output path %r is not exist.", output_model)
        return False

    return True


def is_graph_input_static(graph: BaseGraph) -> bool:
    for input_ in graph.inputs:
        for dim in input_.shape:
            try:
                dim = int(dim)
            except ValueError:
                return False
            if dim <= 0:
                return False
    return True


def list_knowledges():
    registered_knowledges = KnowledgeFactory.get_knowledge_pool()
    logger.info('Available knowledges:')
    for idx, name in enumerate(registered_knowledges):
        logger.info(f'  {idx:2d} {name}')
    for j, name in enumerate(ARGS_REQUIRED_KNOWLEDGES):
        logger.info(f'  {idx+j+1:2d} {name}')


def cli_eva(
    model_path: pathlib.Path,
    optimizer: GraphOptimizer,
    recursive: bool,
    verbose: bool,
    processes: int,
):
    if model_path.is_dir():
        onnx_files = list(model_path.rglob('*.onnx') if recursive else model_path.glob('*.onnx'))
    else:
        onnx_files = [model_path]

    if processes > 1:
        evaluate = partial(evaluate_onnx, optimizer=optimizer, verbose=verbose)
        with Pool(processes) as p:
            res = p.map(evaluate, onnx_files)
        for file, knowledges in zip(onnx_files, res):
            if not knowledges:
                continue
            summary = ','.join(knowledges)
            logger.info(f'{file}\t{summary}')
        return

    for onnx_file in onnx_files:
        knowledges = evaluate_onnx(optimizer=optimizer, model=onnx_file, verbose=verbose)
        if not knowledges:
            continue
        summary = ','.join(knowledges)
        logger.info(f'{onnx_file}\t{summary}')


def optimize_onnx(
    optimizer: Union[GraphOptimizer, list],
    input_model: pathlib.Path,
    output_model: pathlib.Path,
    infer_test: bool,
    config: InferTestConfig,
    big_kernel_config: Optional[BigKernelConfig] = None,
) -> List[str]:
    '''Optimize a onnx file and save as a new file.'''
    try:
        graph = OnnxGraph.parse(input_model.as_posix(), add_name_suffix=False)
    except Exception as exc:
        logger.warning('%s model parse failed.', input_model.as_posix())
        logger.warning('exception: %s', exc)
        return []

    if isinstance(optimizer, list):
        knowledges = [know for know in optimizer if know not in ARGS_REQUIRED_KNOWLEDGES]
        optimizer = GraphOptimizer(knowledges)

    if big_kernel_config:
        optimizer.register_big_kernel(
            graph, big_kernel_config.attention_start_node, big_kernel_config.attention_end_node
        )

    config.is_static = is_graph_input_static(graph)
    if infer_test:
        if not (config.is_static or (config.input_shape_range and config.dynamic_shape and config.output_size)):
            logger.warning('Failed to optimize %s with inference test.', input_model.as_posix())
            logger.warning('Didn\'t specify input_shape_range or dynamic_shape or output_size.')
            return []
        optimize_action = partial(optimizer.apply_knowledges_with_infer_test, cfg=config)
    else:
        optimize_action = optimizer.apply_knowledges

    try:
        graph_opt, applied_knowledges = optimize_action(graph=graph)
    except Exception as exc:
        logger.warning('%s optimize failed.', input_model.as_posix())
        logger.warning('exception: %s', exc)
        return []

    if applied_knowledges:
        if not output_model.parent.exists():
            ms_makedirs(output_model.parent)
        graph_opt.save(output_model.as_posix())
    return applied_knowledges


def evaluate_onnx(
    model: pathlib.Path,
    optimizer: GraphOptimizer,
    verbose: bool,
) -> List[str]:
    '''Search knowledge pattern in a onnx model.'''
    if verbose:
        logger.info(f'Evaluating {model.as_posix()}')
    try:
        graph = OnnxGraph.parse(model.as_posix(), add_name_suffix=False)
    except Exception as exc:
        logger.warning('%s match failed.', model.as_posix())
        logger.warning('exception: %s', exc)
        return []
    try:
        graph, applied_knowledges = optimizer.apply_knowledges(graph)
    except Exception as exc:
        logger.warning('%s match failed.', model.as_posix())
        logger.warning('exception: %s', exc)
        return []
    return applied_knowledges


class FormatMsg:
    def show(self, file=None) -> None:
        logger.error(self.format_message())


def convert_to_graph_optimizer(ctx: click.Context, param: click.Option, value: str) -> GraphOptimizer:
    '''Process and validate knowledges option.'''
    try:
        return GraphOptimizer([v.strip() for v in value.split(',')])
    except Exception as err:
        raise click.BadParameter('No valid knowledge provided!') from err


default_off_knowledges = [
    'KnowledgeEmptySliceFix',
    'KnowledgeTopkFix',
    'KnowledgeGatherToSplit',
    'KnowledgeSplitQKVMatmul',
    'KnowledgeDynamicReshape',
    'KnowledgeResizeModeToNearest',
]


def parse_opt_name(params: click.Option):
    if len(params.opts) == 0:
        opt_name = params.name
    elif len(params.opts) == 1:
        opt_name = params.opts[0]
    else:
        opt_name = "/".join(params.opts)
    return opt_name


def check_args(ctx: click.Context, params: click.Option, value: str):
    """
    check whether the param is provided
    """
    args = [
        opt 
        for param in ctx.command.params 
        for opt in param.opts
    ]
    if value in args:
        opt_name = parse_opt_name(params)
        raise click.BadOptionUsage(option_name=opt_name, message="Option {} requires an argument".format(opt_name))
    return value


def check_node_name(ctx: click.Context, params: click.Option, value: str):
    value = check_args(ctx, params, value)
    args = [
        opt + "=" 
        for param in ctx.command.params 
        for opt in param.opts
    ]
    opt_name = parse_opt_name(params)
    for arg in args:
        if value.startswith(arg):
            raise click.BadOptionUsage(option_name=opt_name, message="Option {} requires an argument".format(opt_name))
    return value


def validate_opt_converter(ctx: click.Context, param: click.Option, value: str) -> str:
    '''Process and validate knowledges option.'''
    if value.lower() not in ['atc']:
        raise click.BadParameter('Invalid converter.')
    return value.lower()
