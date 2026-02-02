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

from __future__ import division
from __future__ import print_function

import os

from collections import Counter
from collections import defaultdict, deque
from itertools import combinations
from datetime import datetime, timezone

import pandas as pd

from components.utils.log import logger
from components.utils.file_open_check import (
    ms_open, 
    OpenException, 
    MAX_SIZE_LIMITE_CONFIG_FILE,
    MAX_SIZE_LIMITE_NORMAL_FILE,
)
from msit_graph.graph_extract.graph_extract import GraphAnalyze
from components.utils.file_open_check import sanitize_cell_for_dataframe


class SimpleNode:
    def __init__(self, name, typename):
        self.name = name
        self.typename = typename
        self.inputs = []
        self.outputs = []


def parse_pbtxt(graph_def):
    nodes = {}
    """
    Parse the graph definition from a .pbtxt file into a dictionary of SimpleNode objects.

    :param graph_def: The parsed GraphDef object from the .pbtxt file.
    :return: A dictionary where keys are node names and values are SimpleNode objects.
    """
    # First pass: Create all nodes and collect their inputs
    for node in graph_def.node:
        node_name = node.name
        node_type = node.op_type
        nodes[node_name] = SimpleNode(node_name, node_type)

        inputs = [input_name.split(':')[0] for input_name in node.input]
        nodes[node_name].inputs = inputs

    # Second pass: Populate outputs based on inputs
    for node_name, node in nodes.items():
        for input_node_name in node.inputs:
            if input_node_name in nodes:
                nodes[input_node_name].outputs.append(node_name)

    return nodes


def bfs_subgraph(root_name, nodes, max_nodes):
    """
    Perform a breadth-first search (BFS) starting from root_name to find a subgraph
    with up to max_nodes nodes.

    :param root_name: The name of the root node to start BFS from.
    :param nodes: Dictionary of SimpleNode objects representing the graph.
    :param max_nodes: Maximum number of nodes in the resulting subgraph.
    :return: List of node names forming the subgraph.
    """
    queue = deque([root_name])
    visited = set()
    subgraph_nodes = []

    while queue and len(subgraph_nodes) < max_nodes:
        current_name = queue.popleft()
        if current_name not in visited:
            visited.add(current_name)
            subgraph_nodes.append(current_name)
            for neighbor_name in nodes[current_name].outputs:
                if neighbor_name not in visited:
                    queue.append(neighbor_name)

    return subgraph_nodes


def get_subtree(nodes, node_name):
    """
    Get all nodes in the subtree rooted at node_name.

    :param nodes: Dictionary of SimpleNode objects representing the graph.
    :param node_name: The name of the root node of the subtree.
    :return: Set of node names in the subtree.
    """
    subtree = set()
    stack = [node_name]
    while stack:
        current = stack.pop()
        if current not in subtree:
            subtree.add(current)
            for output in nodes[current].outputs:
                stack.append(output)
    return subtree


def generate_subgraphs(root_name, nodes, bfs_nodes, min_nodes):
    """
    Generate all possible subgraphs by removing non-root nodes iteratively.

    :param root_name: The name of the root node.
    :param nodes: Dictionary of SimpleNode objects representing the graph.
    :param bfs_nodes: List of node names forming the initial BFS-generated subgraph.
    :return: Set of tuples representing unique subgraph paths.
    """

    if len(bfs_nodes) < min_nodes:
        return None

    subgraphs = defaultdict(set)
    non_root_nodes = set(bfs_nodes) - {root_name}

    # Add the initial BFS subgraph as the first subgraph
    initial_subgraph_types = tuple(nodes[node_name].typename for node_name in bfs_nodes)
    subgraphs[initial_subgraph_types].add(root_name)

    # Generate all subsets of non-root nodes
    for r in range(1, len(non_root_nodes) + 1):
        for subset in combinations(non_root_nodes, r):
            removed_nodes = set(subset)
            # Remove the subtree rooted at each node in the subset
            for node in subset:
                removed_nodes.update(get_subtree(nodes, node))

            # Calculate the remaining nodes in the subgraph while maintaining order
            remaining_nodes = list(filter(lambda node: node not in removed_nodes, bfs_nodes))

            # Ensure the subgraph contains at least two nodes: root and one child
            if len(remaining_nodes) >= min_nodes:
                subgraph_types = tuple(nodes[node_name].typename for node_name in remaining_nodes)
                logger.debug(f"subgraph_types: {subgraph_types}")
                subgraphs[subgraph_types].add(root_name)

    return subgraphs


def find_duplicate_subgraphs(graphs, max_nodes=8, min_nodes=2):
    """
    Find and count all duplicate subgraphs in the given graph.

    :param nodes: Dictionary of SimpleNode objects representing the graph.
    :param subgraph_count: Dictionary to store counts of each subgraph.
    :param max_nodes: Maximum number of nodes in the subgraphs.
    """
    total_count = sum(len(nodes) for nodes in graphs)
    index_counter = 1
    subgraph_count = defaultdict(int)
    subgraph_roots = defaultdict(list)
    for nodes in graphs:
        for node_name in nodes.keys():
            bfs_nodes = bfs_subgraph(node_name, nodes, max_nodes)
            logger.debug(f"------bfs_nodes {bfs_nodes}")
            logger.info(f"processing node {node_name}, {index_counter}/{total_count}")
            subgraphs = generate_subgraphs(node_name, nodes, bfs_nodes, min_nodes)
            if not subgraphs:
                continue
            logger.debug(f"subgraphs:{subgraphs}")
            index_counter += 1
            for subgraph, roots in subgraphs.items():
                subgraph_count[subgraph] += 1
                subgraph_roots[subgraph].extend(roots)

    return subgraph_count, subgraph_roots


def has_subgraph(graph):
    for node in graph.node:
        if node.op_type == "subgraph":
            return True
    return False


def extract_indices(root_nodes_list):
    # Extract indices from a list of node names
    indices = [node.split('_')[-1] for node in root_nodes_list]
    return '; '.join(indices)


def process_contained_subgraphs(duplicate_subgraphs):
    result = []
    current_count = None
    current_group = []

    def contains(small, big):
        return all(item in big for item in small)

    for subgraph, count in duplicate_subgraphs:
        if current_count != count:
            # Process the previous group
            if current_group:
                filtered_group = filter_contained(current_group, contains)
                result.extend(filtered_group)
            current_group = [(subgraph, count)]
            current_count = count
        else:
            current_group.append((subgraph, count))

    # Don't forget to process the last group
    if current_group:
        filtered_group = filter_contained(current_group, contains)
        result.extend(filtered_group)

    return result


def filter_contained(group, contains):
    subgraphs = [item[0] for item in group]
    filtered_group = []
    
    for i, subgraph_i in enumerate(subgraphs):
        if not any(contains(subgraph_i, subgraph_j) for j, subgraph_j in enumerate(subgraphs) if i != j):
            filtered_group.append(group[i])
    
    return filtered_group


def stat_subgraph(input_path, max_nodes=8, min_nodes=2, min_times=1):
    if max_nodes > 10 or max_nodes < min_nodes:
        logger.error(f"max_nodes should be between {min_nodes} and 10, please set it to an appropriate value")
        return None
    if min_nodes < 2:
        logger.warning("min_nodes can not be less than 2, it will be set as default value 2.")
        min_nodes = 2
    if min_times < 1:
        logger.warning("min_times can not be less than 1, it will be set as default value 1.")
        min_times = 1

    graph_def = GraphAnalyze.load_graph_def_from_pbtxt(input_path)
    if graph_def is None:
        logger.error("Failed to parse the pbtxt file.")
        return None

    graphs = []
    if has_subgraph(graph_def):
        for node in graph_def.node:
            if node.op_type == "subgraph":
                for attr in node.attribute:
                    if attr.HasField('g'):  # Check if the attribute has a subgraph
                        nodes = parse_pbtxt(attr.g)
                        if nodes is not None:
                            graphs.append(nodes)
                        else:
                            logger.error("Failed to get nodes information.")
                            return None
    else:
        nodes = parse_pbtxt(graph_def)
        if nodes is not None:
            graphs.append(nodes)
        else:
            logger.error("Failed to get nodes information.")
            return None

    subgraph_count, subgraph_roots = find_duplicate_subgraphs(graphs, max_nodes, min_nodes)

    # Sort duplicate subgraphs by their count in descending order
    duplicate_subgraphs = sorted(subgraph_count.items(), key=lambda x: x[1], reverse=True)

    # Remove subgraphs that appear less than min_times
    duplicate_subgraphs = [(value, count) for value, count in duplicate_subgraphs if count >= min_times]

    # Process duplicate subgraphs to remove contained ones within the same count group
    processed_subgraphs = process_contained_subgraphs(duplicate_subgraphs)

    # Prepare data for DataFrame
    data = {'Subgraph': [value for value, _ in processed_subgraphs],
            'Count': [count for _, count in processed_subgraphs],
            'Root Nodes Index': ['; '.join(map(str, subgraph_roots[value])) for value, _ in processed_subgraphs]}

    # Extract indices from Root Nodes
    data['Root Nodes Index'] = [
        extract_indices(root_nodes_str.split('; ')) 
        for root_nodes_str in data['Root Nodes Index']
    ]

    # Create DataFrame
    df = pd.DataFrame(data)
    return df


def calculate_task_durations(subgraph_tuple, average_durations):
    total_duration = 0.0
    try:
        for op_type in subgraph_tuple:
            avg_duration = average_durations.loc[average_durations['OP Type'] == op_type, 
                'Average Task Duration(us)'].values
            if len(avg_duration) > 0:
                total_duration += avg_duration[0]
            else:
                logger.warning(f"No average duration found for OP Type: {op_type}")
        return total_duration
    except (ValueError, TypeError) as e:
        logger.error(f"Error calculating task durations: {e}")
        return None


def calculate_average_durations(profile_df):
    # Group by 'OP Type' and calculate the mean of 'Task Duration(us)'
    average_durations = profile_df.groupby('OP Type')['Task Duration(us)'].mean().reset_index()
    average_durations.columns = ['OP Type', 'Average Task Duration(us)']
    return average_durations


def preprocess_subgraph(subgraph_tuple):
    # Remove 'ge:' prefix from each OP Type
    cleaned_op_types = tuple(op_type.replace('ge:', '') for op_type in subgraph_tuple)
    return cleaned_op_types


def calculate_sum(args):
    try:
        output_path = args.output
        # Check and set default output path if None
        if output_path is None:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = f"fuse_duration_{timestamp}.csv"
        GraphAnalyze.validate_file_path(args.source, '.pbtxt')
        GraphAnalyze.validate_file_path(args.profile, '.csv')

        subgraph_df = stat_subgraph(args.source, args.max_nodes, args.min_nodes, args.min_times)
        if subgraph_df is None:
            logger.error("Failed to get subgraph data.")
            return

        profile = args.profile
        profile_df = pd.read_csv(profile)
        if profile_df.empty:
            logger.error("Profile DataFrame is empty, maybe %r is missing required columns ." % profile)
            return

        # Preprocess Subgraph column in subgraph_df
        subgraph_df['Subgraph'] = subgraph_df['Subgraph'].apply(preprocess_subgraph)
        if subgraph_df['Subgraph'].isnull().any():
            logger.warning("Preprocessing failed for some subgraphs.")

        # Calculate average durations for each OP Type
        average_durations = calculate_average_durations(profile_df)
        if average_durations is None:
            logger.error("Failed to calculate average durations.")
            return

        # Apply the function to create 'Task Sum Duration(us)' column
        subgraph_df['Task Sum Duration(us)'] = subgraph_df['Subgraph'].apply(
            lambda x: calculate_task_durations(x, average_durations)
        )
        if subgraph_df['Task Sum Duration(us)'].isnull().any():
            logger.warning("Some task duration calculations failed.")

        # Calculate 'Total Duration(us)' column
        subgraph_df['Total Duration(us)'] = subgraph_df['Count'] * subgraph_df['Task Sum Duration(us)']

        # Select only the required columns for the output
        output_df = subgraph_df[
            ['Subgraph', 'Count', 'Root Nodes Index', 'Task Sum Duration(us)', 'Total Duration(us)']
        ]
        sanitize_cell_for_dataframe(output_df)
        # Save the result to output CSV file
        with ms_open(output_path, 'w') as file: 
            output_df.to_csv(file, index=False)
        logger.info("Results saved to %r" % output_path)
    except Exception as e:
        logger.error(f"Unexpected error during calculation: {e}")