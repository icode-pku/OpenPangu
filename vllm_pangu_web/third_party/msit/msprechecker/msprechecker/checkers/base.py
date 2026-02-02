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

from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, auto

from ..presets import RuleManager
from ..utils import (
    Traverser, MacroExpander, ExpandError, Evaluator,
    get_handler, ErrorType, BaseError, ErrorHandler
)
from ..validators import get_validator
from ..collectors import CollectResult
from ..utils import get_npu_count, get_current_ip_and_addr


class WaMLError(Exception):
    pass


class BlockType(Enum):
    BLK_BASIC = auto()
    BLK_COND = auto()
    BLK_SEARCH = auto()
    BLK_UNKNOWN = auto()


class BaseChecker(ABC):
    def __init__(self, *, error_handler: ErrorHandler = None, rule_manager: RuleManager = None):
        self.error_handler = error_handler or get_handler(ErrorType.ERR_CHECK)
        self.rule_manager = rule_manager or RuleManager()

    @abstractmethod
    def _check(self, results):
        pass

    def check(self, collect_result: CollectResult) -> BaseError:
        if not collect_result.error_handler.empty():
            return collect_result.error_handler

        results = collect_result.data
        results = results[0] if isinstance(results, tuple) else results 
        self._check(results)

        return self.error_handler


class NodeChecker(BaseChecker):
    def __init__(self, *, error_handler: ErrorHandler = None, rule_manager: RuleManager = None):
        super().__init__(error_handler=error_handler, rule_manager=rule_manager)
        self.max_nesting_depth = 10

    @staticmethod
    def _get_builtins():
        """
        This method is used to init builtin variables for yaml to use.
        
        For example, yaml can use "${global.cur_ip}" to refer to the current ip.
        Current supports:
            cur_ip: current host ip
            npu_count: number of npu devices
        """
        builtins = {
            "global.cur_ip": get_current_ip_and_addr()[1],
            "global.npu_count": get_npu_count()
        }

        return builtins

    @staticmethod
    def _get_temp_variables(ref_section, visited_nodes):
        """
        This method is used to get temp variables from yaml
        
        We allow yaml to have a "ref" representing variables for later usage.
        """
        temporary_variables = {}
        if not isinstance(ref_section, list):
            raise WaMLError("Reference section must be a list of reference definitions")

        for ref_item in ref_section:
            if 'from' not in ref_item:
                raise WaMLError("Reference definition must contain 'from' field")
            if 'as' not in ref_item:
                raise WaMLError("Reference definition must contain 'as' field for variable naming")

            scope = ref_item['from']
            variable_name = ref_item['as']
            full_var_name = f'ref.{variable_name}'
            # in case scope does not found, we accept a default value, if 'default' not present, set to None
            temporary_variables[full_var_name] = ref_item.get('default')

            if scope not in visited_nodes:
                continue

            if not isinstance(visited_nodes[scope], list):
                temporary_variables[full_var_name] = visited_nodes[scope]
                continue

            attribute = ref_item.get('select')
            condition = ref_item.get('where', True)
            for i, _ in enumerate(visited_nodes[scope]):
                local_path = f"{scope}[{i}]"
                processed_condition = MacroExpander.expand(condition, local_path, visited_nodes)
                condition_result = Evaluator.evaluate(processed_condition)

                if not condition_result:
                    continue

                if attribute is None:
                    processed_attribute = visited_nodes[local_path]
                else:
                    processed_attribute = MacroExpander.expand(attribute, local_path, visited_nodes)

                temporary_variables[full_var_name] = Evaluator.evaluate(processed_attribute)

        return temporary_variables
    
    @abstractmethod
    def _get_rules(self):
        pass

    def _check(self, results) -> ErrorHandler:
        builtin_variables = self._get_builtins()
        visited_nodes = Traverser.traverse(results)
        visited_nodes.update(builtin_variables)

        rules = self._get_rules()
        if 'ref' in rules:
            ref_section = rules.pop('ref')
            temporary_variables = self._get_temp_variables(ref_section, visited_nodes)
            visited_nodes.update(temporary_variables)

        self._validate_nodes(rules, visited_nodes)

        return self.error_handler

    def _validate_nodes(self, rules, visited_nodes):
        queue = deque()
        queue.append((rules, ""))

        while queue:
            node, path = queue.popleft()
            
            if isinstance(node, dict) and 'expected' in node:
                self._validate_expect(node, path, visited_nodes)
            elif isinstance(node, dict):
                self._validate_dict(node, path, queue, visited_nodes)
            elif isinstance(node, list):
                self._validate_list(node, path, queue, visited_nodes)
    
    def _validate_expect(self, node, path, visited_nodes):
        self.max_nesting_depth = 10

        block = node['expected']
        result, reason, severity, suggest = self._process_block(block, path, visited_nodes)
        actual = visited_nodes[path]
        if actual is None:
            actual = "<missing>"

        if not result:
            self.error_handler.add_error(
                path=path, reason=node.get('reason', reason),
                severity=node.get('severity', severity),
                expected=suggest,
                actual=actual
            )

    def _process_block(self, block, path, visited_nodes):
        if self.max_nesting_depth < 0:
            raise WaMLError(f"Maximum nesting depth of {self.max_nesting_depth} exceeded")

        self.max_nesting_depth -= 1

        block_type = self._get_block_type(block)        
        if block_type == BlockType.BLK_BASIC:
            return self._process_basic_block(block, path, visited_nodes)
        elif block_type == BlockType.BLK_SEARCH:
            return self._process_search_block(block, path, visited_nodes)
        elif block_type == BlockType.BLK_COND:
            return self._process_conditional_block(block, path, visited_nodes)
        else:
            if "type" not in block or "value" not in block:
                raise WaMLError(f"Unknown block type: {block_type}")
            
            case_block = {"case": block, "reason": "", "severity": "high"}
            return self._process_basic_block(case_block, path, visited_nodes)

    def _get_block_type(self, block):
        if isinstance(block, dict):
            if all(keyword in block for keyword in ('case', 'reason')):
                return BlockType.BLK_BASIC

            if all(keyword in block for keyword in ('if', 'then')):
                return BlockType.BLK_COND

            if all(keyword in block for keyword in ('contains_all',)):
                return BlockType.BLK_SEARCH

        return BlockType.BLK_UNKNOWN

    def _process_search_block(self, block, path, visited_nodes):
        actual = visited_nodes[path]

        if not isinstance(actual, list):
            raise WaMLError(f"Expected a list for search operation, got {type(actual).__name__}")
    
        results = []
        if 'len' in block:
            expected_length = block['len']
            reason = f"{path} 元素个数和 'worldSize' 不一致"
            severity = "high"
            suggest = expected_length

            if isinstance(expected_length, int):
                result = len(actual) == expected_length
            elif isinstance(expected_length, str):
                expected_expr = MacroExpander.expand(expected_length, path, visited_nodes)
                evaluated_expr = Evaluator.evaluate(expected_expr)
                result = len(actual) == evaluated_expr
                suggest = evaluated_expr
            else:
                raise TypeError(
                    f"Expected 'len' to be int or basic block, got {type(expected_length).__name__} instead"
                )

            if not result:
                self.error_handler.add_error(
                    path=path, reason=reason, severity=severity,
                    expected=list(range(suggest)), actual=actual
                )

        search_items = block['contains_all']
        if not search_items:
            return True, "", "", ""  # 空搜索列表视为成功

        for item in search_items:
            found = False
            for i, _ in enumerate(visited_nodes[path]):
                local_path = f"{path}[{i}]"
                try:
                    result, reason, severity, suggest = self._process_block(
                        block=item, path=local_path, visited_nodes=visited_nodes
                    )
                except Exception:
                    result = False

                self.max_nesting_depth += 1
                if result:  # 只要有一个匹配就成功
                    found = True
                    break

            if not found:
                results.append((
                    item['reason'], item.get(severity, 'high'),
                    item.get('suggest', item['case'])
                ))

        for reason, severity, suggest in results:
            self.error_handler.add_error(
                path=path, reason=reason, severity=severity,
                expected=suggest, actual="<missing>"
            )

        return True, "", "", "" # 返回 True 外面不需要再 add_error

    def _process_conditional_block(self, block, path, visited_nodes):
        condition = block['if']
        expanded_condition = MacroExpander.expand(condition, path, visited_nodes)
        condition_result = Evaluator.evaluate(expanded_condition)

        next_block = block['then'] if condition_result else block.get('else')

        if next_block is None:
            return True, "", "", ""  # 默认返回成功

        result = self._process_block(block=next_block, path=path, visited_nodes=visited_nodes)
        return result

    def _process_basic_block(self, block, path, visited_nodes):
        case = block['case']
        reason = block['reason']
        severity = block.get('severity', 'high')

        actual = visited_nodes[path]
        if isinstance(case, dict):
            if 'type' not in case or 'value' not in case:
                raise WaMLError("Basic block must contain 'type' and 'value' fields if 'case' if a dict type")

            expected_value = case['value']
            expanded_value = MacroExpander.expand(expected_value, path, visited_nodes)
            evaluated_value = Evaluator.evaluate(expanded_value)
            validator = get_validator(case['type'])
            
            return validator.validate(actual, evaluated_value), reason, severity, evaluated_value

        if case == "absent" and actual is not None:
            return actual is not None, reason, severity, None
        if case == "present" and actual is None:
            return actual is None, reason, severity, "-"
        if case == "alert":
            return False, reason, "medium", "-"

        expanded_expr = MacroExpander.expand(case, path, visited_nodes)
        evaluated_expr = Evaluator.evaluate(expanded_expr)
        if not evaluated_expr:
            return evaluated_expr, reason, severity, block.get('suggest', case)

        return True, "", "", ""

    def _validate_dict(self, node: dict, path: str, queue: deque, visited_nodes: dict):
        for k, v in node.items():
            new_path = k if not path else f"{path}.{k}"
            if new_path not in visited_nodes:
                new_path += '%'
                visited_nodes[new_path] = None
            queue.append((v, new_path))

    def _validate_list(self, node: list, path: str, queue: deque, visited_nodes: dict):
        for i, v in enumerate(node):
            new_path = f"[{i}]" if not path else f"{path}[{i}]"
            if new_path not in visited_nodes:
                new_path += '%'
                visited_nodes[new_path] = None
            queue.append((v, new_path))
