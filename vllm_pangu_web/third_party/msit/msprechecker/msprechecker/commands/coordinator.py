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

import os
import json
import argparse
from typing import List

import yaml
from msguard.security import open_s

from .base import CommandType
from .legacy import show_legacy_warnings
from .banner import BannerPresenter
from ..collectors import ConfigCollector, CollectResult
from ..reporters import Reporter
from ..presets import RuleManager
from .base import CommandStrategy, CommandType
from ..collectors import (
    BaseCollector,
    EnvCollector,
    SysCollector,
    ConfigCollector,
    AscendCollector,
    HCCLCollector,
    PingCollector,
    WeightCollector,
    CPUStressCollector,
    NPUStressCollector,
    UserConfigCollector,
    MindIEEnvCollector,
    ModelConfigCollector,
    MIESConfigCollector,
    TlsCollector,
    VnicCollector,
    LinkCollector,
)
from ..checkers import (
    UserConfigChecker,
    MindIEEnvChecker,
    ModelConfigChecker,
    EnvChecker,
    SysChecker,
    AscendChecker,
    HCCLChecker,
    StressChecker,
    PDChecker,
    MIESConfigChecker,
    TlsChecker,
    VnicChecker,
    LinkChecker,
    PingChecker,
)
from ..comparators import Comparator
from ..reporters import Reporter
from ..utils import (
    FrameworkType, ParserRegistry, update_model_type,
    CheckErrorHandler, ConfigErrorHandler, global_logger, singleton
)


class CollectorFactory:
    @staticmethod
    def create(args: argparse.Namespace):
        default_collectors = [
            SysCollector(),
            AscendCollector(),
        ]  # all scenes applies
        if getattr(args, 'framework', 'mindie') == 'vllm' or \
            getattr(args, 'command') == CommandType.CMD_DUMP:
            default_collectors.append(EnvCollector(filter_env=getattr(args, 'filter_env', False)))

        special_collectors = CollectorFactory.dispatch_collectors_by_scene(args)
        extra_collectors = CollectorFactory.dispatch_extra_collectors(args)

        return list(set(default_collectors + special_collectors + extra_collectors))

    @staticmethod
    def dispatch_collectors_by_scene(args: argparse.Namespace) -> List[BaseCollector]:
        collectors = []

        # 大 EP
        if getattr(args, "user_config_path", None) or getattr(args, "mindie_env_path", None):
            if getattr(args, "user_config_path", None):
                collectors.append(UserConfigCollector(config_path=args.user_config_path))
            if getattr(args, "mindie_env_path", None):
                collectors.append(MindIEEnvCollector(config_path=args.mindie_env_path))

        # PD Disaggregation (single container)

        # PD Mix
        elif getattr(args, "mies_config_path", None):
            collectors.append(EnvCollector(filter_env=getattr(args, 'filter_env', False)))
            collectors.append(MIESConfigCollector(config_path=args.mies_config_path))

        return collectors

    @staticmethod
    def dispatch_extra_collectors(args: argparse.Namespace) -> List[BaseCollector]:
        collectors = []

        if getattr(args, "rank_table_path", None):
            if not getattr(args, "framework", None):
                global_logger.warning(
                    "Passing '--rank-table-path' without providing '--scene', "
                    "msprechecker cannot determine the exact framework type of the rank table. "
                    "Will use 'mindie' as the default framework."
                )
                args.framework = FrameworkType.TP_MINDIE

            framework_type = FrameworkType(args.framework)
            rank_table_parser = ParserRegistry.get(framework_type)()  # create parser instance
            rank_table = rank_table_parser.parse(args.rank_table_path)

            collectors.extend(
                (
                    PingCollector(rank_table=rank_table),
                    TlsCollector(),
                    HCCLCollector(rank_table=rank_table),
                    LinkCollector(),
                    VnicCollector(),
                )
            )

        if getattr(args, "weight_dir", None):
            model_config_path = os.path.join(args.weight_dir, "config.json")
            collectors.append(ModelConfigCollector(config_path=model_config_path))

            if getattr(args, "command", None) == CommandType.CMD_DUMP:
                chunk_size = getattr(args, 'chunk_size', 32)
                chunk_size *= 1024 ** 2
                collectors.append(WeightCollector(weight_dir=args.weight_dir, chunk_size=chunk_size))

        if getattr(args, "hardware", False):
            collectors.extend((CPUStressCollector(), NPUStressCollector()))

        return collectors


@singleton
class CheckerFactory:
    def __init__(self):
        self._registry = {}
        self._init()

    @staticmethod
    def default_param_extractor(args, collect_result):
        return {
            "rule_manager": RuleManager(
                scene=args.scene, framework=args.framework, custom_rule_path=args.custom_config_path
            ),
            "error_handler": CheckErrorHandler(severity=args.severity_level),
        }

    @staticmethod
    def config_param_extractor(args, collect_result):
        data, file_lines, key_mapping, context_hierarchy = collect_result.data
        return {
            "rule_manager": RuleManager(
                scene=args.scene, framework=args.framework, custom_rule_path=args.custom_config_path
            ),
            "error_handler": ConfigErrorHandler(args.severity_level, file_lines, key_mapping, context_hierarchy),
        }

    @staticmethod
    def stress_param_extractor(args, collect_result):
        return {
            "rule_manager": RuleManager(
                scene=args.scene, framework=args.framework, custom_rule_path=args.custom_config_path
            ),
            "error_handler": CheckErrorHandler(severity=args.severity_level),
            "threshold": getattr(args, "threshold", None),
        }

    def register(self, collector_class, checker_cls, param_extractor=None) -> None:
        param_extractor = param_extractor or self.default_param_extractor
        self._registry[collector_class] = (checker_cls, param_extractor)

    def create(self, collector_cls, args, collect_result):
        if collector_cls not in self._registry:
            raise KeyError(f"No checker registered for collector: {collector_cls.__name__}")

        checker_cls, param_extractor = self._registry[collector_cls]
        params = param_extractor(args, collect_result)
        return checker_cls(**params)

    def _init(self):
        self.register(EnvCollector, EnvChecker)
        self.register(SysCollector, SysChecker)
        self.register(AscendCollector, AscendChecker)
        self.register(HCCLCollector, HCCLChecker)
        self.register(CPUStressCollector, StressChecker, self.stress_param_extractor)
        self.register(NPUStressCollector, StressChecker, self.stress_param_extractor)
        self.register(UserConfigCollector, UserConfigChecker, self.config_param_extractor)
        self.register(MindIEEnvCollector, MindIEEnvChecker, self.config_param_extractor)
        self.register(ModelConfigCollector, ModelConfigChecker, self.config_param_extractor)
        self.register(MIESConfigCollector, MIESConfigChecker, self.config_param_extractor)
        self.register(TlsCollector, TlsChecker)
        self.register(VnicCollector, VnicChecker)
        self.register(LinkCollector, LinkChecker)
        self.register(PingCollector, PingChecker)


class PrecheckStrategy(CommandStrategy):
    @staticmethod
    def execute_pd_disagg(args):
        rule_manager = RuleManager(scene=args.scene, custom_rule_path=args.custom_config_path)
        reporter = Reporter()

        paths_to_find = rule_manager.get_rules().keys()

        collect_data = {}
        for path in paths_to_find:
            if "ref" in path:
                continue
            if os.path.isabs(path):
                global_logger.warning("unsafe, key should not be abspath: {path!r}")
                continue
            full_path = os.path.join(args.config_parent_dir, path)
            load_fn = json.load if full_path.endswith(".json") else lambda f: list(yaml.safe_load_all(f))
            try:
                with open_s(full_path) as f:
                    data = load_fn(f)
            except Exception as e:
                global_logger.error("missing file: %r", full_path)
                return 1

            collect_data[path] = data

        error_handler = CheckErrorHandler(severity=args.severity_level, type_="PD Disaggregation")
        collect_result = CollectResult(collect_data, error_handler)
        checker = PDChecker(rule_manager=rule_manager, error_handler=error_handler)
        check_result = checker.check(collect_result)
        reporter.report(check_result)
        return 0

    @staticmethod
    def execute(args: argparse.Namespace) -> int:
        if args.scene and "pd_disaggregation" in args.scene:
            if not args.config_parent_dir:
                global_logger.error(
                    "Passing '--scene' without providing '--config-parent-dir' will not take any effect!"
                )
            return PrecheckStrategy.execute_pd_disagg(args)

        if args.scene and "," in args.scene:
            parts = args.scene.split(",", 1)
            if len(parts) != 2 or not all(parts):
                global_logger.error("Invalid scene format! Use 'framework,scene'")
                return 1
            args.framework = parts[0].strip()
            args.scene = parts[1].strip()
        else:
            args.framework = args.scene
            args.scene = "default"

        reporter = Reporter()
        collectors = CollectorFactory.create(args)
        checker_factory = CheckerFactory()

        for collector in collectors:
            collect_result = collector.collect()
            checker = checker_factory.create(collector.__class__, args, collect_result)
            check_result = checker.check(collect_result)
            reporter.report(check_result)

        return 0


class DumpStrategy(CommandStrategy):
    @staticmethod
    def execute(args: argparse.Namespace) -> int:
        collectors = CollectorFactory.create(args)

        dump_content = {}
        for collector in collectors:
            collect_result = collector.collect()
            if not collect_result.error_handler.empty():
                DumpStrategy._display_collect_warning(collect_result.error_handler)
                continue

            collect_type = collect_result.error_handler.type
            data = collect_result.data

            if isinstance(collector, ConfigCollector):
                data, _, _, _ = collect_result.data

            dump_content[collect_type] = data

        with open_s(args.output_path, "w") as f:
            json.dump(dump_content, f, indent=4)

            global_logger.info(
                "All information has been saved in: %r. You can use '--output-path' to specify the save location.",
                args.output_path,
            )
            global_logger.info(
                "What's Next?\n\t"
                "You may now use 'msprechecker compare' to compare two or more dumped files for discrepancies!"
            )

        return 0

    @staticmethod
    def _display_collect_warning(error_handler):
        for error in error_handler:
            context = error.context
            global_logger.warning("Error occured while collecting '%s': %s", error_handler.type, context.what)


class CompareStrategy(CommandStrategy):
    @staticmethod
    def execute(args: argparse.Namespace) -> int:
        if len(args.dumped_path) < 2:
            global_logger.error("You need two or more files to compare!")
            return 1

        path_to_data = CompareStrategy._load_dumped_files(args.dumped_path)
        reporter = Reporter()

        comparator = Comparator()
        reporter.report(comparator.compare(path_to_data))
        return 0

    @staticmethod
    def _load_dumped_files(file_paths):
        """Load dumped JSON files for comparison"""
        path_to_data = {}

        for path in file_paths:
            with open_s(path) as f:
                path_to_data[path] = json.load(f)

        return path_to_data


class CommandStrategyFactory:
    def __init__(self) -> None:
        self._registry = {
            CommandType.CMD_PRECHECK: PrecheckStrategy,
            CommandType.CMD_DUMP: DumpStrategy,
            CommandType.CMD_COMPARE: CompareStrategy,
        }

    def register(self, cmd_type, strategy_class) -> None:
        self._registry[cmd_type] = strategy_class

    def create_strategy(self, cmd: CommandType) -> CommandStrategy:
        if cmd not in self._registry:
            raise ValueError(f"No strategy registered for command: {cmd}")

        return self._registry[cmd]()


class Coordinator:
    def __init__(self) -> None:
        self._strategy_factory = CommandStrategyFactory()

    def execute(self, parser: argparse.ArgumentParser) -> int:
        """Execute the appropriate action based on command"""
        args = parser.parse_args()
        update_model_type(args)
        show_legacy_warnings(args)

        cmd = getattr(args, "command", None)
        if not cmd:
            BannerPresenter().print_banner()
            parser.print_help()
            return 1

        args.command = CommandType(cmd)
        strategy = self._strategy_factory.create_strategy(args.command)
        return strategy.execute(args)
