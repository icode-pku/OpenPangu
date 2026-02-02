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

import pathlib

import click

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.common.click_utils import (
    convert_to_graph_optimizer,
    default_off_knowledges,
    validate_opt_converter,
)


opt_optimizer = click.option(
    '-k',
    '--knowledges',
    'optimizer',
    default=','.join(
        knowledge
        for knowledge in KnowledgeFactory.get_knowledge_pool().keys()
        if knowledge not in default_off_knowledges
    ),
    type=str,
    callback=convert_to_graph_optimizer,
    help='Knowledges(index/name) you want to apply. Seperate by comma(,).',
)


opt_processes = click.option(
    '-p',
    '--processes',
    'processes',
    default=1,
    type=click.IntRange(1, 64),
    help='Use multiprocessing in evaluate mode, determine how many processes should be spawned.',
)


opt_verbose = click.option(
    '-v', '--verbose', 'verbose', is_flag=True, default=False, help='Show progress in evaluate mode.'
)


opt_recursive = click.option(
    '-r',
    '--recursive',
    'recursive',
    is_flag=True,
    default=False,
    help='Process onnx in a folder recursively if any folder provided as PATH.',
)


arg_output = click.argument('output_model', nargs=1, type=click.Path(path_type=pathlib.Path))


arg_input = click.argument(
    'input_model',
    nargs=1,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=pathlib.Path),
)


arg_start = click.argument(
    'start_node_names',
    type=click.STRING,
)


arg_end = click.argument(
    'end_node_names',
    type=click.STRING,
)


opt_check = click.option(
    '-c',
    '--is-check-subgraph',
    'is_check_subgraph',
    is_flag=True,
    default=False,
    help='Whether to check subgraph.',
)


arg_path = click.argument(
    'path', nargs=1, type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True, path_type=pathlib.Path)
)


opt_device = click.option(
    '-d', '--device', 'device', default=0, type=click.IntRange(min=0), help='Device_id.'
)


opt_loop = click.option(
    '-l',
    '--loop',
    'loop',
    default=100,
    type=click.IntRange(min=1),
    help='How many times to run the test inference.',
)


opt_soc = click.option(
    '-s', '--soc', 'soc', default='Ascend310P3', type=str, help='Soc_version.'
)


opt_converter = click.option(
    '-c',
    '--converter',
    'converter',
    default='atc',
    type=str,
    callback=validate_opt_converter,
    help='OM Converter.',
)


opt_threshold = click.option(
    '--threshold',
    'threshold',
    default=0,
    type=click.FloatRange(min=-1),
    help='Threshold of inference speed improvement,'
    'knowledges with less improvement won\'t be used.'
    'Can be a negative number, which means accept'
    'negative optimization.',
)


opt_infer_test = click.option(
    '-t',
    '--infer-test',
    'infer_test',
    is_flag=True,
    default=False,
    help='Run inference to determine whether to apply knowledges optimization.',
)


opt_big_kernel = click.option(
    '-bk',
    '--big-kernel',
    'big_kernel',
    is_flag=True,
    default=False,
    help='Whether to apply big kernel optimize knowledge.',
)


opt_attention_start_node = click.option(
    '-as',
    '--attention-start-node',
    'attention_start_node',
    type=str,
    default="",
    help='Start node of the first attention block, it must be set when apply big kernel knowledge.',
)


opt_attention_end_node = click.option(
    '-ae',
    '--attention-end-node',
    'attention_end_node',
    type=str,
    default="",
    help='End node of the first attention block, it must be set when apply big kernel knowledge.',
)


opt_input_shape = click.option(
    '--input-shape',
    'input_shape',
    type=str,
    help='Input shape of onnx graph.',
)


opt_input_shape_range = click.option(
    '--input-shape-range', 'input_shape_range', type=str, help='Specify input shape range for OM converter.'
)


opt_dynamic_shape = click.option(
    '--dynamic-shape', 'dynamic_shape', type=str, help='Specify input shape for dynamic onnx in inference.'
)


opt_output_size = click.option('--output-size', 'output_size', type=str, help='Specify real size of graph output.')


opt_subgraph_input_shape = click.option(
    '-sis', '--subgraph_input_shape', 'subgraph_input_shape', type=str, help='Specify the input shape of subgraph'
)


opt_subgraph_input_dtype = click.option(
    '-sit', '--subgraph_input_dtype', 'subgraph_input_dtype', type=str, help='Specify the input dtype of subgraph'
)
