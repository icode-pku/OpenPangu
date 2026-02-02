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
import pathlib

from components.utils.parser import BaseCommand
from auto_optimizer.graph_optimizer.optimizer import (
    GraphOptimizer,
    InferTestConfig,
    BigKernelConfig,
    ARGS_REQUIRED_KNOWLEDGES,
)
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from components.debug.common import logger
from auto_optimizer.common.click_utils import (
    optimize_onnx,
    list_knowledges,
    cli_eva,
    check_input_path,
    check_output_model_path,
)
from auto_optimizer.common.args_check import (
    check_in_model_path_legality,
    check_out_model_path_legality,
    check_soc,
    check_range,
    check_min_num_1,
    check_min_num_2,
    check_shapes_string,
    check_dtypes_string,
    check_io_string,
    check_nodes_string,
    check_single_node_string,
    check_normal_string,
    check_shapes_range_string,
    check_ints_string,
    check_path_string,
    check_in_path_legality,
)
from auto_optimizer.common.click_utils import default_off_knowledges
from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory


class ListCommand(BaseCommand):
    def handle(self, _):
        list_knowledges()


class EvaluateCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            '--path',
            required=True,
            type=check_in_path_legality,
            help='Target onnx file or directory containing onnx file',
        )
        parser.add_argument(
            '-know',
            '--knowledges',
            default=','.join(
                knowledge
                for knowledge in KnowledgeFactory.get_knowledge_pool().keys()
                if knowledge not in default_off_knowledges
            ),
            type=check_normal_string,
            help='Knowledges(index/name) you want to apply. Seperate by comma(,).',
        )
        parser.add_argument(
            '-r',
            '--recursive',
            action="store_true",
            default=False,
            help='Process onnx in a folder recursively if any folder provided \
                            as PATH.',
        )
        parser.add_argument(
            '-v',
            '--verbose',
            action="store_true",
            default=False,
            help='Show progress in evaluate mode.',
        )
        parser.add_argument(
            '-p',
            '--processes',
            default=1,
            type=check_range,
            help='Use multiprocessing in evaluate mode, \
                            determine how many processes should be spawned.',
        )

    def handle(self, args):
        if not check_input_path(args.path):
            return

        path_ = pathlib.Path(args.path)
        knowledge_list = [v.strip() for v in args.knowledges.split(',')]
        filtered_knowledge_list = []
        for know in knowledge_list:
            if know in ARGS_REQUIRED_KNOWLEDGES:
                logger.warning(f"Knowledge {know} cannot be evaluate")
            else:
                filtered_knowledge_list.append(know)

        if not filtered_knowledge_list:
            return
        optimizer = GraphOptimizer(filtered_knowledge_list)
        cli_eva(path_, optimizer, args.recursive, args.verbose, args.processes)


class OptimizeCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            '-in',
            '--input',
            dest='input_model',
            required=True,
            type=check_in_model_path_legality,
            help='Input onnx model to be optimized',
        )
        parser.add_argument(
            '-of',
            '--output-file',
            dest='output_model',
            required=True,
            type=check_out_model_path_legality,
            help='Output onnx model name',
        )
        parser.add_argument(
            '-know',
            '--knowledges',
            default=','.join(
                knowledge
                for knowledge in KnowledgeFactory.get_knowledge_pool().keys()
                if knowledge not in default_off_knowledges
            ),
            type=check_nodes_string,
            help='Knowledges(index/name) you want to apply. Seperate by comma(,).',
        )
        parser.add_argument(
            '-t',
            '--infer-test',
            action="store_true",
            default=False,
            help='Run inference to determine whether to apply knowledges \
                            optimization.',
        )
        parser.add_argument(
            '-bk',
            '--big-kernel',
            action="store_true",
            default=False,
            help='Whether to apply big kernel optimize knowledge.',
        )
        parser.add_argument(
            '-as',
            '--attention-start-node',
            type=check_single_node_string,
            default="",
            help='Start node of the first attention block, \
                            it must be set when apply big kernel knowledge.',
        )
        parser.add_argument(
            '-ae',
            '--attention-end-node',
            type=check_single_node_string,
            default="",
            help='End node of the first attention block, \
                            it must be set when apply big kernel knowledge.',
        )
        parser.add_argument(
            '-soc',
            '--soc-version',
            dest='soc',
            default='Ascend310P3',
            type=check_normal_string,
            help='Soc_version.',
        )
        parser.add_argument('-d', '--device', default=0, type=check_soc, help='Device_id.')
        parser.add_argument(
            '--loop',
            default=100,
            type=check_min_num_1,
            help='How many times to run the test inference.',
        )
        parser.add_argument(
            '--threshold',
            default=0,
            type=check_min_num_2,
            help='Threshold of inference speed improvement,'
            'knowledges with less improvement won\'t be used.'
            'Can be a negative number, which means accept'
            'negative optimization',
        )
        parser.add_argument(
            '-is',
            '--input-shape',
            type=check_shapes_string,
            help='Input shape of onnx graph.',
        )
        parser.add_argument(
            '--input-shape-range', type=check_shapes_range_string, help='Specify input shape range for OM converter.'
        )
        parser.add_argument(
            '--dynamic-shape', type=check_shapes_string, help='Specify input shape for dynamic onnx in inference.'
        )
        parser.add_argument(
            '-outsize', '--output-size', type=check_ints_string, help='Specify real size of graph output.'
        )

    def handle(self, args):
        if not check_input_path(args.input_model) or not check_output_model_path(args.output_model):
            return

        # compatibility for click < 8.0
        input_model_ = pathlib.Path(os.path.abspath(args.input_model))
        output_model_ = pathlib.Path(os.path.abspath(args.output_model))
        if input_model_ == output_model_:
            logger.warning('output_model is input_model, refuse to overwrite origin model!')
            return

        knowledge_list = [v.strip() for v in args.knowledges.split(',')]
        for know in knowledge_list:
            if not args.big_kernel and know in ARGS_REQUIRED_KNOWLEDGES:
                knowledge_list.remove(know)
                logger.warning("Knowledge {} cannot be ran when close big_kernel config.".format(know))

        if not knowledge_list:
            return

        if args.big_kernel:
            big_kernel_config = BigKernelConfig(
                attention_start_node=args.attention_start_node, attention_end_node=args.attention_end_node
            )
        else:
            big_kernel_config = None

        config = InferTestConfig(
            converter='atc',
            soc=args.soc,
            device=args.device,
            loop=args.loop,
            threshold=args.threshold,
            input_shape=args.input_shape,
            input_shape_range=args.input_shape_range,
            dynamic_shape=args.dynamic_shape,
            output_size=args.output_size,
        )
        applied_knowledges = optimize_onnx(
            optimizer=knowledge_list,
            input_model=input_model_,
            output_model=output_model_,
            infer_test=args.infer_test,
            config=config,
            big_kernel_config=big_kernel_config,
        )
        if args.infer_test:
            logger.info('=' * 100)
        if applied_knowledges:
            logger.info('Result: Success')
            logger.info('Applied knowledges: ')
            for knowledge in applied_knowledges:
                logger.info(f'  {knowledge}')
            logger.info(f'Path: {input_model_} -> {output_model_}')
        else:
            logger.info('Result: Unable to optimize, no knowledges matched.')
        if args.infer_test:
            logger.info('=' * 100)


class ExtractCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            '-in',
            '--input',
            dest='input_model',
            required=True,
            type=check_in_model_path_legality,
            help='Input onnx model to be optimized',
        )
        parser.add_argument(
            '-of',
            '--output-file',
            dest='output_model',
            required=True,
            type=check_out_model_path_legality,
            help='Output onnx model name',
        )
        parser.add_argument(
            '-snn', '--start-node-names', required=False, type=check_nodes_string, help='The names of start nodes'
        )
        parser.add_argument(
            '-enn', '--end-node-names', required=False, type=check_nodes_string, help='The names of end nodes'
        )
        parser.add_argument(
            '-ck',
            '--is-check-subgraph',
            action="store_true",
            default=False,
            help='Whether to check subgraph.',
        )
        parser.add_argument(
            '-sis', '--subgraph-input-shape', type=check_shapes_string, help='Specify the input shape of subgraph'
        )
        parser.add_argument(
            '-sit', '--subgraph-input-dtype', type=check_dtypes_string, help='Specify the input dtype of subgraph'
        )

    def handle(self, args):
        if not check_input_path(args.input_model) or not check_output_model_path(args.output_model):
            return

        input_model_ = pathlib.Path(os.path.abspath(args.input_model))
        output_model_ = pathlib.Path(os.path.abspath(args.output_model))
        if input_model_ == output_model_:
            logger.warning('output_model is input_model, refuse to overwrite origin model!')
            return

        # parse start node names and end node names
        if args.start_node_names:
            start_node_names = [node_name.strip() for node_name in args.start_node_names.split(',')]
        else:
            start_node_names = None

        if args.end_node_names:
            end_node_names = [node_name.strip() for node_name in args.end_node_names.split(',')]
        else:
            end_node_names = None

        onnx_graph = OnnxGraph.parse(args.input_model)
        try:
            onnx_graph.extract_subgraph(
                start_node_names,
                end_node_names,
                args.output_model,
                args.is_check_subgraph,
                args.subgraph_input_shape,
                args.subgraph_input_dtype,
            )
        except ValueError as err:
            logger.error(err)


class ConcatenateCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            '-g1',
            '--graph1',
            required=True,
            type=check_in_model_path_legality,
            help='First onnx model to be consolidated',
        )
        parser.add_argument(
            '-g2',
            '--graph2',
            required=True,
            type=check_in_model_path_legality,
            help='Second onnx model to be consolidated',
        )
        parser.add_argument(
            '-io',
            '--io-map',
            required=True,
            type=check_io_string,
            help='Pairs of output/inputs representing outputs \
                            of the first graph and inputs of the second graph to be connected',
        )
        parser.add_argument(
            '-pref',
            '--prefix',
            dest='graph_prefix',
            required=False,
            type=check_path_string,
            default='pre_',
            help='Prefix added to all names in a graph',
        )
        parser.add_argument(
            '-cgp',
            '--combined-graph-path',
            default='',
            type=check_out_model_path_legality,
            help='Output combined onnx graph path',
        )

    def handle(self, args):
        onnx_graph1 = OnnxGraph.parse(args.graph1)
        onnx_graph2 = OnnxGraph.parse(args.graph2)

        # parse io_map
        # out0:in0;out1:in1...
        io_map_list = []
        for pair in args.io_map.strip().split(";"):
            if not pair:
                continue
            out, inp = pair.strip().split(",")
            io_map_list.append((out, inp))

        try:
            combined_graph = OnnxGraph.concat_graph(
                onnx_graph1, onnx_graph2, io_map_list, prefix=args.graph_prefix, graph_name=args.combined_graph_path
            )
        except Exception as err:
            logger.error(err)

        if args.combined_graph_path:
            combined_graph_path = args.combined_graph_path
        else:
            combined_graph_path = args.graph1[:-5] + "_" + args.graph2[:-5] + ".onnx"
            combined_graph_path = combined_graph_path.replace("/", "_")

        try:
            combined_graph.save(combined_graph_path)
        except Exception as err:
            logger.error(err)

        logger.info(f"Combined ONNX model saved in: {combined_graph_path}")


def get_cmd_instance():
    surgeon_help_info = "surgeon tool for onnx modifying functions."
    list_cmd_instance = ListCommand("list", "List available Knowledges")
    evaluate_cmd_instance = EvaluateCommand(
        "evaluate", "Evaluate model matching specified knowledges", alias_name="eva"
    )
    optimize_cmd_instance = OptimizeCommand("optimize", "Optimize model with specified knowledges", alias_name="opt")
    extract_cmd_instance = ExtractCommand("extract", "Extract subgraph from onnx model", alias_name="ext")
    concatenate_cmd_instance = ConcatenateCommand(
        "concatenate", "Concatenate two onnxgraph into combined one onnxgraph", alias_name="concat"
    )
    return BaseCommand(
        "surgeon",
        surgeon_help_info,
        [
            list_cmd_instance,
            evaluate_cmd_instance,
            optimize_cmd_instance,
            extract_cmd_instance,
            concatenate_cmd_instance,
        ],
    )
