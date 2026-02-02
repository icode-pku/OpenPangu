# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import ast
import json
import argparse
from typing import List

from msguard import validate_args, Rule
from msguard.security import open_s

from .parser import Parser
from .data_source import DataSource
from .util import load, cmate_logger
from ._test import make_test_suite, RuleTestRunner
from .visitor import MetaVisitor, RuleVisitor, PrettyFormatter, SetEnvGenerator, RequirementGenerator


def _parse_configs(configs: List[str]):
    res = {}
    if not configs:
        return res
    
    splitter = ':'
    parse_splitter = '@'

    for entry in configs:
        parts = entry.split(splitter, 1)

        if len(parts) != 2:
            cmate_logger.error(
                "Expected '<name>:<path>' or '<name>:<path>@<parse_type>' where:\n",
                "    <name>: special identifier defined in the cmate file",
                "    <path>: the file path corresponding to the <name> in the file system",
                "    <parse_type>: (optional) defines how to parse this config path, e.g. json, yaml, etc"
            )
            raise ValueError

        name = parts[0]
        fields = parts[1].split(parse_splitter, 1)
        path, parse_type = fields[0], None if len(fields) == 1 else fields

        res[name] = (path, parse_type)

    return res


def _parse_contexts(configs: List[str]):
    res = {}
    if not configs:
        return res
    
    splitter = ':'
    for entry in configs:
        parts = entry.split(splitter, 1)
        
        if len(parts) != 2:
            cmate_logger.error(
                "Expected '<name>:<value>' where:\n",
                "    <name>: special context namepsace defined in the cmate file\n",
                "    <value>: the value to <name>.\n"
                "Note that 2 will be considered as integer 2, and for string 2 you need to pass '2'",
            )
            raise ValueError

        name, value = parts
        try:
            res[name] = ast.literal_eval(value)
        except ValueError:
            res[name] = value

    return res


def _display_text(result):
    meta = result['metadata']
    required = result['requirements']

    cmate_logger.info("\n[Overview]")
    if meta:
        for key, value in meta.items():
            if value:
                cmate_logger.info(f"  {key}: {value}")
    else:
        cmate_logger.info('  No information found')
    
    if not required:
        cmate_logger.info("\nNo validation rules available.")
        return
    
    cmate_logger.info(f"\n[Available Rules: {len(required)}]")
    
    for i, (rule_name, rule_info) in enumerate(required.items(), 1):
        desc = rule_info['desc'] or 'No description provided'
        parse_type = rule_info['parse_type'] or 'unknown'
        cmate_logger.info(f"\n{i}. %s - %s (%s)", rule_name, desc, parse_type)
        contexts = rule_info['contexts']
        if contexts:
            cmate_logger.info(f"   Contexts:")
            for ctx_name, ctx_info in contexts.items():
                ctx_desc = ctx_info['desc']
                ctx_values = ctx_info['possible_values']
                
                cmate_logger.info(f"     - {ctx_name}")
                if ctx_desc:
                    cmate_logger.info(f"         {ctx_desc}")
                cmate_logger.info(f"         Values: {ctx_values}")
    cmate_logger.info(f"\nUsage: -c <rule_name>:<file_path> -C <context>:<value>")


def _display_json(result):
    cmate_logger.info(json.dumps(result, indent=4))


def _validate_and_load_dependencies(requirements, input_configs, input_contexts):
    input_configs = input_configs or {}
    input_contexts = input_contexts or {}

    data_source = DataSource()
    matched_name = []

    for name, (input_path, input_parse_type) in input_configs.items():
        if name not in requirements:
            cmate_logger.warning('no such rule: %s', name)
            continue

        matched_name.append(name)
        # user parse type or rule parse type or None (ext)
        parse_type = input_parse_type or requirements[name]['parse_type'] 
        try:
            parsed_data = load(input_path, parse_type)
        except OSError:
            cmate_logger.error(
                "Config file '%s' not found at path: '%s'. Please verify the file exists and path is correct.",
                name, input_path
            )
            return False, data_source
        except TypeError:
            cmate_logger.error(
                "Unsupported parse type '%s' for config '%s' at: '%s'.\n"
                "Solution: Specify a supported parse type using: '-c %s:%s@<parse-type>'",
                parse_type, name, input_path, name, input_path
            )
            return False, data_source
        except Exception as e:
            cmate_logger.error(
                "Failed to parse config '%s' from: '%s'. The %s syntax appears invalid: %s",
                name, input_path, parse_type, e
            )
            return False, data_source
        data_source.flatten(name, parsed_data)

    if not matched_name:
        cmate_logger.error('%s', list(requirements.keys()))
        return False, DataSource

    for name in matched_name:
        contexts = requirements[name]['contexts']
        missing_contexts = contexts.copy()

        for ctx_name, ctx_value in input_contexts.items():
            if ctx_name not in contexts:
                cmate_logger.warning('no such context required: %s', ctx_name)
                continue
            data_source[f'context::{ctx_name}'] = ctx_value
            missing_contexts.pop(ctx_name)

        if missing_contexts:
            cmate_logger.error('%s', missing_contexts)
            return False, data_source

    return True, data_source


def _collect_only(ruleset):
    pretty_formatter = PrettyFormatter()

    msg = []
    for namespace in ruleset:
        msg.append(f'<Namespace {namespace}>')
        for rule_node in ruleset[namespace]:
            rule = pretty_formatter.format(rule_node)
            msg.append(f'  <Rule-{rule_node.lineno} {rule}>')
    cmate_logger.info('\n'.join(msg))
    return 0


def _actual_run(ruleset, data_source, failfast, verbosity):
    test_runner = RuleTestRunner(
        failfast=failfast,
        verbosity=verbosity
    )

    test_suite = make_test_suite(data_source, ruleset)
    result = test_runner.run(test_suite)
    return result.wasSuccessful()


def parse(rule_path):
    parser = Parser()

    with open_s(rule_path) as f:
        node = parser.parse(f.read())

    return node


def inspect(rule_path: str, output_format: str):
    output_format_map = {
        'text': _display_text,
        'json': _display_json
    }

    node = parse(rule_path)

    meta_visitor = MetaVisitor()

    res = {
        'metadata': meta_visitor.visit(node),
        'requirements': RequirementGenerator().generate(node)
    }

    if output_format not in output_format_map:
        raise ValueError('Unsupported output format: %r' % output_format)

    return output_format_map[output_format](res)


def run(
    rule_path: str, configs=None, contexts=None,
    failfast=False, verbosity=False, collect_only=False, severity='info'
):
    node = parse(rule_path)
    requirements = RequirementGenerator().generate(node)

    ret, data_source = _validate_and_load_dependencies(requirements, configs, contexts)
    if not ret:
        return 1
    
    rule_visitor = RuleVisitor(data_source, severity)

    try:
        ruleset = rule_visitor.visit(node)
    except KeyError as e:
        cmate_logger.error(
            "Rule collection failed: %s. " \
            "Most likely because you are trying to use a value from a namespace " \
            "that was not specified in the dependencies section, nor in the partition section.",
            e
        )
        return 1

    SetEnvGenerator(data_source).generate(node)

    if collect_only:
        return _collect_only(ruleset)
    
    return _actual_run(ruleset, data_source, failfast, verbosity)


def main():
    arg_parser = argparse.ArgumentParser(prog='cmate')
    subparsers = arg_parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run', help='Run rule validations')
    run_parser.add_argument('rule', type=validate_args(Rule.input_file_read), help='Rule file to validate')
    run_parser.add_argument(
        '--contexts', '-C', nargs='*', help='Context variables in name:path format (or name:path@type).'
    )
    run_parser.add_argument(
        '--configs', '-c', nargs='*',
        help='Config files in name:path format (or name:path@type). Use @json or @yaml to force parse type'
    )
    run_parser.add_argument(
        '-co', '--collect-only', action='store_true', help='Show which rules will run, do not execute'
    )
    run_parser.add_argument(
        '-x', '--fail-fast', action='store_true', dest='failfast', help='Stop on first failure/error'
    )
    run_parser.add_argument(
        '-v', '--verbose', action='store_true', dest='verbose', help='Verbose: show each test name and result'
    )
    run_parser.add_argument(
        '-s', '--severity', 
        choices=['info', 'warning', 'error'],
        default='info',
        help='Minimum severity to run: info (all), warning (no info), error (errors only)'
    )

    inspect_parser = subparsers.add_parser('inspect', help='Inspect rule requirements')
    inspect_parser.add_argument('rule', type=validate_args(Rule.input_file_read), help='Rule file to inspect')
    inspect_parser.add_argument('--format', '-f', choices=['text', 'json'], default='text', help='Output format')

    args = arg_parser.parse_args()

    if args.command == 'inspect':
        return inspect(args.rule, args.format)

    elif args.command == 'run':
        configs = _parse_configs(args.configs)
        contexts = _parse_contexts(args.contexts)
        return run(args.rule, configs, contexts, args.failfast, args.verbose, args.collect_only, args.severity)

    return 0
