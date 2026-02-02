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

from auto_optimizer.graph_refactor.onnx import OnnxNode, OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.knowledges.big_kernel.attention_parser import AttentionParser
from auto_optimizer.pattern.knowledges.big_kernel.transform_refactor import TransformRefactor
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.pattern.pattern import MatchPattern
from components.debug.common import logger
from auto_optimizer.pattern.knowledges.big_kernel.util import gen_normal_subgraph
from components.utils.util import safe_get


layernorm_pattern = Pattern() \
    .add_node("start", ['Add', 'Mul', 'Transpose']) \
    .add_node('mean1', ['ReduceMean']) \
    .add_node('sub', ['Sub']) \
    .add_edge('start', 'mean1') \
    .add_edge('start', 'sub') \
    .add_edge('mean1', 'sub') \
    .add_node('pow', ['Pow']) \
    .add_edge('sub', 'pow') \
    .add_node('mean2', ['ReduceMean']) \
    .add_edge('pow', 'mean2') \
    .add_node('add1', ['Add']) \
    .add_edge('mean2', 'add1') \
    .add_node('sqrt', ['Sqrt']) \
    .add_edge('add1', 'sqrt') \
    .add_node('div', ['Div']) \
    .add_edge('sub', 'div') \
    .add_node('mul', ['Mul']) \
    .add_edge('div', 'mul') \
    .add_node('add2', ['Add']) \
    .add_edge('mul', 'add2') \
    .set_loop(MatchPattern.MATCH_ONCE_OR_MORE)


class KnowledgeBigKernel(KnowledgeBase):
    def __init__(self, graph, start_node, end_node):
        super(KnowledgeBigKernel, self).__init__()
        self.attention_pattern = self.get_pattern(graph, start_node, end_node)
        self.end_node_type = graph.get_node(end_node, node_type=OnnxNode).op_type
        self._register_apply_funcs(self.attention_pattern, [self.big_kernel_apply])
        self.attention_idx = 0
        self.attention_ori_shape = None

    def get_pattern(self, graph, start_node, end_node):
        subgraph = graph.extract_subgraph([start_node], [end_node])
        pattern = Pattern()

        for node in subgraph.nodes:
            pattern.add_node(node.name, [node.op_type])
            for inp in node.inputs:
                prev_node = subgraph.get_prev_node(inp)
                if prev_node and prev_node.name in pattern.node_dict:
                    pattern.add_edge(prev_node.name, node.name)
        pattern.set_loop(MatchPattern.MATCH_ONCE_OR_MORE)
        return pattern

    def big_kernel_apply(self, graph: OnnxGraph, match_result: MatchResult):
        logger.info("Start to optimize {} attention in graph.".format(self.attention_idx))
        refactor = TransformRefactor(graph)
        match_nodes = {
            safe_get(node, 0).name: safe_get(node, 0)
            for _, node in safe_get(match_result.node_dicts, 0).items()
        }
        atten_start_node, atten_end_node, softmax = refactor.get_anchor_nodes(match_nodes, self.end_node_type)
        if not atten_start_node or not atten_end_node or not softmax:
            raise ValueError("Cann\'t get attention start node or softmax node or attention end node.")

        attention = refactor.graph.extract_subgraph([atten_start_node.name], [atten_end_node.name])
        attention_parser = AttentionParser(graph=attention, start_node=atten_start_node,
                                           end_node=atten_end_node,
                                           softmax=softmax)
        attention_parser.parse_graph()

        prefix = str(self.attention_idx) + "."
        normal_subgraph = gen_normal_subgraph(prefix)

        refactor.update_normal_subgraph(normal_subgraph, attention_parser, prefix)

        refactor.replace_subgraph(normal_subgraph, attention_parser, prefix)

        refactor.update_graph(attention_parser, normal_subgraph)

        self.attention_ori_shape = attention_parser.params.get("ori_shape")
        self.attention_idx += 1

        return True

    def post_process(self, graph: OnnxGraph):
        refactor = TransformRefactor(graph)
        refactor.graph.update_map()

        layernorm_result = self.search_subgraph(graph, layernorm_pattern)
        refactor.update_layernorm(layernorm_result, self.attention_ori_shape)
        refactor.graph.update_map()
        refactor.graph.toposort()
        try:
            refactor.infer_graph_shape()
        except Exception as exp:
            logger.warning("Infer shape fail, exp: {}".format(exp))
        refactor.update_mask_add_node(self.attention_idx)
        refactor.remove_unused_initializers()
        return True

