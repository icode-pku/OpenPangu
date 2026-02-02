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

import argparse

from components.utils.parser import BaseCommand
from components.utils.file_open_check import FileStat
from components.utils.log import set_log_level, LOG_LEVELS
from msit_graph.graph_extract.graph_extract import GraphAnalyze
from msit_graph.subgraph_stat.subgraph_stat import calculate_sum
from msit_graph.inspect.scan import execute


LOG_LEVELS_LOWER = [ii.lower() for ii in LOG_LEVELS.keys()]


def check_output_path_legality(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except FileNotFoundError as ffe:
        raise argparse.ArgumentTypeError("output path %r does not exist." % path_value) from ffe
    except PermissionError as pe:
        raise argparse.ArgumentTypeError("permission denied for output path %r." % path_value) from pe
    except Exception as err:
        raise argparse.ArgumentTypeError(
            "an unexpected error occurred while checking the output path %r." % path_value
            ) from err
    if not file_stat.is_basically_legal("write", strict_permission=False):
        raise argparse.ArgumentTypeError("output path %r cannot be written to." % path_value)
    return path_value


def check_input_path_legality(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except FileNotFoundError as ffe:
        raise argparse.ArgumentTypeError("input path %r does not exist." % path_value) from ffe
    except PermissionError as pe:
        raise argparse.ArgumentTypeError("permission denied for input path %r." % path_value) from pe
    except Exception as err:
        raise argparse.ArgumentTypeError(
            "an unexpected error occurred while checking the input path %r." % path_value
            ) from err
    if not file_stat.is_basically_legal('read', strict_permission=False):
        raise argparse.ArgumentTypeError("input path %r cannot be read." % path_value)
    return path_value


class StatsCommand(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:

        parser.add_argument(
            "-i",
            "--input",
            dest="input",
            type=check_input_path_legality,
            required=True,
            help="input pbtxt path.E.g:--input /xx/xxxx/xx.pbtxt"
        )
        parser.add_argument("-l", "--log-level", dest="log_level", default="info", 
                            choices=LOG_LEVELS_LOWER, help="specify log level")
    
    def handle(self, args, **kwargs) -> None:
        set_log_level(args.log_level)
        GraphAnalyze.print_graph_stat(args.input)


class StripCommand(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:

        parser.add_argument(
            "-i",
            "--input",
            type=check_input_path_legality,
            required=True,
            help="input pbtxt path.E.g:--input /xx/xxxx/xx.pbtxt"
        )
        parser.add_argument(
            "--level",
            type=int,
            default=3,
            choices=[1, 2, 3],
            help="Strip the redundant information on the graph to open the graph faster. "
                "Choices are: \n"
                "1 - Most detailed mode, only delete Const node and Data Node\n"
                "2 - Moderate detail mode, delete Const node, Data Node, and the attribute(unless shape) of nodes\n"
                "3 - Most brief mode (default), delete Const node, Data Node, and the attribute of nodes"
        )
        parser.add_argument(
            "-o",
            '--output',
            required=False,
            type=check_output_path_legality,
            help="output pbtxt path.E.g:--output /xx/xxxx/xx.pbtxt"
        )
        parser.add_argument("-l", "--log-level", dest="log_level", default="info", 
                            choices=LOG_LEVELS_LOWER, help="specify log level")

    def handle(self, args, **kwargs) -> None:
        set_log_level(args.log_level)
        GraphAnalyze.strip(args.input, args.level, args.output)

        
class ExtractCommand(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:
        parser.add_argument(
            "-i",
            "--input",
            type=check_input_path_legality,
            required=True,
            help="input pbtxt path.E.g:--input /xx/xxxx/xx.pbtxt"
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            help="output pbtxt path.E.g:--output /xx/xxxx/xx.pbtxt"
        )
        parser.add_argument(
            "--start-node",
            default=None,
            help="Extract subgraphs by range, from start node to end node.. E.g: --start-node node_name"
        )
        parser.add_argument(
            "--end-node",
            default=None,
            help="Extract subgraphs by range, from start node to end node. E.g: --end-node node_name"
        )
        parser.add_argument(
            "--center-node",
            default=None,
            help="Extracts subgraphs with the node as the center. E.g: --center-node node_name"
        )
        parser.add_argument(
            "--layer-number",
            type=int,
            default=1,
            help="front and back layers. E.g: --layer-number 2"
        )
        parser.add_argument(
            "--only-forward",
            action='store_true',
            help="only dump nodes forward. E.g: --only-forward"
        )
        parser.add_argument(
            "--only-backward",
            action='store_true',
            help="only dump nodes backward. E.g: --only-backward"
        )
        parser.add_argument(
            "--without-leaves",
            action='store_true',
            help="Without leaves when generate the result graph. E.g: --without-leaves"
        )
        parser.add_argument(
            "--stop-name",
            default=None,
            action="append",
            help="Specify a node name which stop the extract iteration. E.g: --stop-name node_name"
        )
        parser.add_argument("-l", "--log-level", dest="log_level", default="info", 
                            choices=LOG_LEVELS_LOWER, help="specify log level")

    def handle(self, args, **kwargs) -> None:
        set_log_level(args.log_level)
        GraphAnalyze.extract_sub_graph(args)


class FuseCommand(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:

        parser.add_argument(
            "-s",
            "--source",
            type=check_input_path_legality,
            required=True,
            help="input pbtxt path.E.g:--source /xx/xxxx/xx.pbtxt"
        )
        parser.add_argument(
            "-p",
            "--profile",
            type=check_input_path_legality,
            required=True,
            help="input profiling path.E.g:--profile /xx/xxxx/xx.csv"
        )
        parser.add_argument(
            "--max-nodes",
            type=int,
            default=8,
            help="Limiting the maximum number of nodes contained in a repeated subgraph"
        )
        parser.add_argument(
            "--min-nodes",
            type=int,
            default=2,
            help="Limiting the minimum number of nodes contained in a repeated subgraph"
        )
        parser.add_argument(
            "--min-times",
            type=int,
            default=1,
            help="Filter repeated subgraphs whose occurrence times are less than it."
        )          
        parser.add_argument(
            "-o",
            '--output',
            required=False,
            type=check_output_path_legality,
            help='output pbtxt path.E.g:--output /xx/xxxx/xx.csv'
        )
        parser.add_argument("-l", "--log-level", dest="log_level", default="info", 
                            choices=LOG_LEVELS_LOWER, help="specify log level")

    def handle(self, args, **kwargs) -> None:
        set_log_level(args.log_level)
        calculate_sum(args)


class InspectCommand(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:
        parser.add_argument(
            "-i",
            "--input",
            type=check_input_path_legality,
            required=True,
            help="<str> input pbtxt path.E.g:--input /xx/xxxx/xx.pbtxt"
        )
        parser.add_argument(
            "-t", 
            "--type", 
            dest="type", 
            type=str, 
            required=True, 
            choices=["dshape"], 
            help="<str> The scan currently supports only dynamic shape inspection."
        )
        parser.add_argument("-l", "--log-level", dest="log_level", default="info", 
                            choices=LOG_LEVELS_LOWER, help="specify log level")
        parser.add_argument(
            "-o", 
            "--output", 
            dest="output", 
            type=check_output_path_legality, 
            default="./", 
            help="<str> A directory path, generate a table of dynamic shape operators with headers: Op_name, "
            "Input_name, and Output_name."
        )

    def handle(self, args, **kwargs) -> None:
        set_log_level(args.log_level)
        execute(args)


def get_cmd_instance():
    graph_analyze_help_info = "Graph analyze Tools."
    stats_cmd_instance = StatsCommand("stats", "Print statistic operator infomation")
    strip_cmd_instance = StripCommand("strip", """Strip the redundant information on the pbtxt to 
        open the graph faster. This could be useful if the pbtxt has a huge size that can not 
        even be loaded."""
    )
    extract_cmd_instance = ExtractCommand("extract", """Extract subgraph. There are two extraction modes: 
        center diffusion and start-end.  The center diffusion mode is as follows: Specify one or more nodes as 
        the diffusion center, and then dump the multi-layer nodes upward or downward. The start-end mode is 
        to specify one or more groups of start nodes and end nodes, and dump all nodes between the start and 
        end nodes."""
    )
    fuse_cmd_instance = FuseCommand("fuse", "Count the number of repeated subgraphs and these average duration.")
    inspect_cmd_instance = InspectCommand("inspect", "Scan the .pbtxt graph to obtain dynamic shape operators.")

    instances = [
        stats_cmd_instance, strip_cmd_instance, extract_cmd_instance, fuse_cmd_instance, inspect_cmd_instance
    ]
    return BaseCommand("graph", graph_analyze_help_info, instances)
