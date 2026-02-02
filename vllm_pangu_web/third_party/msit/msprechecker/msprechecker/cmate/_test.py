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

import sys
import time
import shutil
import unittest
from typing import Set

from .visitor import PrettyFormatter, Evaluator
from . import _ast


class RuleAssertionError(AssertionError):
    def __init__(self, rule_node, history):
        super().__init__()
        self.rule_node = rule_node
        self.history = history

        self.pretty_formatter = PrettyFormatter()

    def build_err_msg(self):
        indent = len(self.rule_node.severity.value) + 2
        lines = []

        test = self.pretty_formatter.visit(self.rule_node.test)
        lines.append('>'.ljust(indent) + test)
        lines.append('')
        for k, v in self.history:
            if isinstance(v, tuple):
                line = f'{k} -> {v[1]} -> {v[0]}'
            else:
                line = f'{k} -> {v}'
            
            lines.append('E'.ljust(indent) + line)
        lines.append('')
        lines.append(f'{self.rule_node.severity} {self.rule_node.msg}')
        return '\n'.join(lines)


class RuleTestResult(unittest.TestResult):
    def __init__(self, stream=None, descriptions=None, verbosity=1, total_cnt=None, failfast=False):
        super().__init__()
        self.stream = stream
        self.seen = set()
        self.total_cnt = total_cnt
        self.cols, _ = shutil.get_terminal_size()
        self.status_chars = []
        self.on_sub_test = False
        self.verbosity = verbosity or 1
        self.failfast = failfast
        self.shouldStop = False

    def startTestRun(self):
        self.stream.writeln()

    def stopTestRun(self):
        self.on_sub_test = False
        self.stream.writeln()

    def startTest(self, test):
        super().startTest(test)
        namespace = test.namespace
        # Only print per-scope compact header when not in verbose mode.
        if namespace not in self.seen:
            self.seen.add(namespace)
            if getattr(self, 'verbosity', 1) == 1:
                header = f"{namespace}: "
                self.stream.write(header)
                self.status_chars.clear()
                self.stream.flush()

    def addError(self, test, err):
        super().addError(test, err)
        # replace stored tuple with exc_value for nicer formatting later
        if self.errors:
            last = self.errors.pop()
            exc_val = last[1][1] if isinstance(last[1], tuple) and len(last[1]) >= 2 else last[1]
            self.errors.append((last[0], exc_val))
        # verbosity: print immediate line
        if self.verbosity > 1:
            # print pytest-like verbose line
            self._print_verbose_line(test, 'ERROR')
        else:
            self._update(test, 'E')

        # fail-fast support
        if self.failfast:
            self.shouldStop = True

    def addFailure(self, test, err):
        super().addFailure(test, err)
        # change to store test instance and err instance instead
        self.failures.pop()
        self.failures.append((test, err[1]))
        if self.verbosity > 1:
            self._print_verbose_line(test, 'FAILED')
        else:
            self._update(test, 'F')
            
        if self.failfast:
            self.shouldStop = True

    def addSuccess(self, test):
        super().addSuccess(test)
        if self.verbosity > 1:
            self._print_verbose_line(test, 'PASSED')
        else:
            self._update(test, '.')

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            # include reason in parentheses but keep left width reasonable
            label = 'SKIPPED'
            self._print_verbose_line(test, label)
        else:
            self._update(test, 's')

    def addExpectedFailure(self, test, err):
        self._update(test, 'x')
        return super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self._update(test, 'U')
        return super().addUnexpectedSuccess(test)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)

        if self.on_sub_test: # sub test is a test that will add testsRun outside
            self.total_cnt += 1
            self.testsRun += 1
        self.on_sub_test = True

        if err is None:
            self._update(test, '.')
        elif issubclass(err[0], test.failureException):
            self._update(test, 'F')
            # change to store test instance and err instance instead
            self.failures.pop()
            self.failures.append((test, err[1]))
            if self.verbosity > 1:
                self.stream.writeln('')
                self.stream.writeln(f"{test.id()} ... \033[31mFAILED\033[0m (subtest)")
        else:
            self._update(test, 'E')
            if self.verbosity > 1:
                self.stream.writeln('')
                self.stream.writeln(f"{test.id()} ... \033[31mERROR\033[0m (subtest)")

    def printErrors(self):
        if self.errors:
            self.stream.writeln('\033[31m' + ' ERRORS '.center(self.cols, '=') + '\033[0m')
            for test, exc in self.errors:
                self.stream.writeln('\033[31m' + f' {test.id()} '.center(self.cols, '_') + '\033[0m')
                self.stream.writeln()
                self.stream.write(str(exc))
        if self.failures:
            self.stream.writeln(' FAILURES '.center(self.cols, '='))
            for test, err in sorted(self.failures, key=lambda item: item[1].rule_node.severity, reverse=True):
                self.stream.writeln('\033[31m' + f' {test.id()} '.center(self.cols, '_') + '\033[0m')
                self.stream.writeln()
                self.stream.writeln(err.build_err_msg())
        self.stream.writeln()
        self.stream.flush()
    
    def _update(self, test, ch):
        namespace = test.namespace
        # append raw char
        self.status_chars.append(ch)
        # prepare colored display mapping but compute widths from raw chars
        color_map = {
            '.': '\033[32m.\033[0m',
            'F': '\033[31mF\033[0m',
            'E': '\033[31mE\033[0m',
            's': '\033[33ms\033[0m',
            'x': 'x',
            'U': 'U'
        }
        status_raw = ''.join(self.status_chars)
        status_disp = ''.join(color_map.get(c, c) for c in self.status_chars)

        # 计算百分比
        total = self.total_cnt
        pct = int(100 * (self.testsRun / total))
        pct_s = f"[{pct:3d}%]"
        
        # 构建输出行
        header = f"{namespace}: "
        # use raw lengths for layout (colored sequences are invisible width-wise)
        visible_len = len(header) + len(status_raw)
        terminal_width = self.cols
        available_space = terminal_width - visible_len - len(pct_s)

        if getattr(self, 'verbosity', 1) == 1:
            if available_space > 0:
                pad = ' ' * available_space
                line = f"{header}{status_disp}{pad}{pct_s}"
            else:
                # if line too long, truncate raw status and recompute display
                overflow = -available_space
                if overflow >= len(self.status_chars):
                    trimmed_disp = ''
                else:
                    trimmed_disp = ''.join(color_map.get(c, c) for c in self.status_chars[:-overflow])
                line = f"{header}{trimmed_disp} {pct_s}"

            # output and flush the compact progress line
            self.stream.write('\r' + line)
            self.stream.flush()
    
    def _print_verbose_line(self, test, status_word):
        """Print a pytest-like verbose line: <testid> <STATUS> [  X%]

        We compute visible widths using plain text (no ANSI) so the percent
        column always aligns and we avoid terminal width jitter caused by
        color escape sequences.
        """
        # recompute terminal width in case it changed
        cols, _ = shutil.get_terminal_size((80, 20))
        total = self.total_cnt or 1
        pct = int(100 * (self.testsRun / total))
        pct_s = f"[{pct:3d}%]"

        tid = test.id()
        # reserve 1 space before pct
        avail = cols - len(pct_s) - 1
        if avail < 10:
            avail = 10

        # keep status word intact at right side; truncate test id if needed
        status_len = len(status_word)
        # one space between tid and status
        reserved = status_len + 1
        avail_for_tid = max(0, avail - reserved)

        tid_part = tid
        if len(tid_part) > avail_for_tid:
            # truncate tid and append ellipsis
            if avail_for_tid <= 3:
                tid_part = tid_part[:avail_for_tid]
            else:
                tid_part = tid_part[: avail_for_tid - 3] + '...'

        left_text = f"{tid_part} {status_word}"
        pad = avail - len(left_text)
        if pad < 0:
            pad = 0

        # color only the status word; build colored left and append computed spaces
        color_map = {
            'PASSED': '\033[32mPASSED\033[0m',
            'FAILED': '\033[31mFAILED\033[0m',
            'ERROR': '\033[31mERROR\033[0m',
            'SKIPPED': '\033[33mSKIPPED\033[0m',
        }
        colored_status = color_map.get(status_word, status_word)
        colored_left = f"{tid_part} {colored_status}"

        line = colored_left + (' ' * pad) + ' ' + pct_s
        self.stream.writeln(line)
        self.stream.flush()


class RuleTestRunner(object):
    rescls = RuleTestResult

    def __init__(self, stream=None, rescls=None, failfast=False, verbosity=False):
        if stream is None:
            stream = sys.stderr
        self.stream = unittest.runner._WritelnDecorator(stream)

        if rescls is not None:
            self.rescls = rescls

        self.verbosity = verbosity
        self.failfast = failfast

    def run(self, suite):
        total = suite.countTestCases()
        res = self.rescls(stream=self.stream, verbosity=self.verbosity, total_cnt=total, failfast=self.failfast)
        
        self.stream.writeln(f"\033[1;38mcollected {total} items\033[0m")
        cols, _ = shutil.get_terminal_size((80, 20))
        if total == 0:
            self.stream.writeln()
            self.stream.writeln('\033[33m' + ' no tests ran in 0.00s '.center(cols, '=') + '\033[0m')
            self.stream.flush()
            return res

        t0 = time.perf_counter()
        start_run = getattr(res, "startTestRun", None)
        if start_run is not None:
            start_run()
        try:
            suite(res)
        finally:
            stop_run = getattr(res, "stopTestRun", None)
            if stop_run is not None:
                stop_run()
        elapsed = time.perf_counter() - t0

        res.printErrors()

        cols, _ = shutil.get_terminal_size()
        summary_parts = []
        if res.errors:
            summary_parts.append(f"{len(res.errors)} errors")
        if res.failures:
            summary_parts.append(f"{len(res.failures)} failed")
        if res.expectedFailures:
            summary_parts.append(f"{len(res.expectedFailures)} expected failures")
        if res.unexpectedSuccesses:
            summary_parts.append(f"{len(res.unexpectedSuccesses)} unexpected successes")
        if res.skipped:
            summary_parts.append(f"{len(res.skipped)} skipped")

        num_passed = (
            res.testsRun
            - len(res.errors)
            - len(res.failures)
            - len(res.expectedFailures)
            - len(res.unexpectedSuccesses)
            - len(res.skipped)
        )
        summary = (" " + ", ".join(summary_parts) + (", " if summary_parts else "") +
                   (f"{num_passed} passed" if num_passed else "no passed") + f" in {elapsed:.3f}s ")

        if num_passed == res.testsRun:
            self.stream.writeln('\033[33m' + summary.center(cols, '=') + '\033[0m')
        else:
            self.stream.writeln('\033[31m' + summary.center(cols, '=') + '\033[0m')
        self.stream.flush()
        return res


def make_test_suite(data_source, ruleset):
    test_suite = unittest.TestSuite()
    test_loader = unittest.TestLoader()

    extra_attrs = dict(evaluator=Evaluator(data_source), data_source=data_source)
    for namespace in ruleset:
        test_case_cls_name = f'Test-{namespace}'
        rule_nodes = ruleset[namespace]
        test_case_cls = _make_test_case(test_case_cls_name, rule_nodes)

        # add extra attrs to test cls
        extra_attrs['namespace'] = namespace
        for name, value in extra_attrs.items():
            setattr(test_case_cls, name, value)

        suite = test_loader.loadTestsFromTestCase(test_case_cls)
        test_suite.addTest(suite)

    return test_suite


def _make_test_case(cls_name, rule_nodes: Set[_ast.Rule]):
    cls_dict = {
        f"test_{rule_node.lineno}": _make_test_method(rule_node)
        for rule_node in rule_nodes
    }
    test_case_cls = type(cls_name, (unittest.TestCase,), cls_dict)

    return test_case_cls


def _make_test_method(rule_node: _ast.Rule):
    def test(inst):
        if not inst.evaluator.eval(rule_node.test):
            history = inst.evaluator.history
            raise RuleAssertionError(rule_node, history)

    return test
