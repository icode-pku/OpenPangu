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
import sys
from .logger import logger
from .utils import find_config_path, load_yaml_config, auto_detect_v1_default
from .symbol_watcher import SymbolWatchFinder


class ServiceProfiler:
    """服务分析器主类。
    
    该类负责初始化和管理整个服务分析器系统，包括：
    - 配置文件加载
    - 版本检测
    - hooker 导入
    - symbol 监听器初始化
    - 传统 hooks 应用
    
    Attributes:
        _hooks_applied (bool): hooks 是否已应用
        _symbol_watcher (SymbolWatchFinder): symbol 监听器
        _vllm_use_v1 (str): vLLM 版本标识
    """
    
    def __init__(self):
        """初始化服务分析器。"""
        self._hooks_applied = False
        self._symbol_watcher = None
        self._vllm_use_v1 = ServiceProfiler._detect_vllm_version()

    @staticmethod
    def _detect_vllm_version() -> str:
        """检测 vLLM 版本。
        
        Returns:
            str: vLLM 版本标识
        """
        env_v1 = os.environ.get('VLLM_USE_V1')
        return env_v1 if env_v1 is not None else auto_detect_v1_default()
    
    @staticmethod
    def _load_config():
        """加载配置文件。
        
        Returns:
            Optional[Dict]: 配置数据，失败时返回 None
        """
        default_cfg = find_config_path()

        env_path = os.environ.get('PROFILING_SYMBOLS_PATH')
        if env_path and str(env_path).lower().endswith(('.yaml', '.yml')):
            # 环境变量目标文件已存在：直接加载
            if os.path.isfile(env_path):
                logger.debug(f"Loading profiling symbols from env path: {env_path}")
                return load_yaml_config(env_path)

            # 目标文件不存在：若有默认配置，尝试复制填充
            if default_cfg:
                try:
                    parent_dir = os.path.dirname(env_path) or '.'
                    os.makedirs(parent_dir, exist_ok=True)
                    with open(default_cfg, 'r', encoding='utf-8') as src, \
                         open(env_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
                    logger.debug(f"Wrote profiling symbols to env path: {env_path}")
                    return load_yaml_config(env_path)
                except Exception as e:
                    logger.warning(f"Failed to write profiling symbols to env path {env_path}: {e}")
            else:
                logger.warning("No default config file found to populate PROFILING_SYMBOLS_PATH")
        elif env_path and not str(env_path).lower().endswith(('.yaml', '.yml')):
            logger.warning(f"PROFILING_SYMBOLS_PATH is not a yaml file: {env_path}")

        # 回退：按默认查找顺序加载
        if not default_cfg:
            logger.warning("No config file found")
            return None
        return load_yaml_config(default_cfg)
    
    def initialize(self):
        """初始化服务分析器。
        
        执行完整的初始化流程：
        1. 检查环境变量
        2. 加载配置文件
        3. 导入内置 hookers
        4. 初始化 symbol 监听器
        """
        try:
            # 检查是否启用了打点
            if not os.environ.get('SERVICE_PROF_CONFIG_PATH'):
                logger.debug("SERVICE_PROF_CONFIG_PATH not set, skipping hooks")
                return
                
            logger.debug("Initializing service profiler")
            
            # 加载配置文件
            config_data = self._load_config()
            if not config_data:
                logger.warning("No configuration loaded, skipping profiler initialization")
                return
            
            # 按版本导入内置 hookers
            self._import_hookers()
            
            # 初始化 symbol 监听器
            self._init_symbol_watcher(config_data)
            
            self._hooks_applied = True
            logger.debug("Service profiler initialized successfully")
            
        except Exception as e:
            logger.exception("Failed to initialize service profiler: %s", str(e))
            self._hooks_applied = False
    
    def _import_hookers(self):
        """按版本导入内置 hookers。
        
        根据 vLLM 版本导入相应的内置 hooker 模块。
        """
        if self._vllm_use_v1 == "0":
            logger.debug("Initializing service profiler with vLLM V0 interface")
            from .vllm_v0 import batch_hookers, kvcache_hookers, model_hookers, request_hookers
        elif self._vllm_use_v1 == "1":
            logger.debug("Initializing service profiler with vLLM V1 interface")
            from .vllm_v1 import batch_hookers, kvcache_hookers, meta_hookers, model_hookers, request_hookers
        else:
            logger.error(f"unknown vLLM interface version: VLLM_USE_V1={self._vllm_use_v1}")
            return
    
    def _init_symbol_watcher(self, config_data):
        """初始化 symbol 监听器。
        
        Args:
            config_data: 配置数据
        """
        self._symbol_watcher = SymbolWatchFinder()
        self._symbol_watcher.load_symbol_config(config_data)
        
        # 安装到 sys.meta_path
        sys.meta_path.insert(0, self._symbol_watcher)
        logger.debug("Symbol watcher installed")
        
        # 检查目标模块是否已经被导入，如果是则立即应用 hooks
        self._check_and_apply_existing_modules()
    
    def _check_and_apply_existing_modules(self):
        """检查目标模块是否已经被导入，如果是则立即应用 hooks。
        
        遍历所有配置的 symbol，检查对应的模块是否已经加载，
        如果是则立即应用相应的 hooks。
        """
        
        logger.debug("Checking for already loaded modules...")
        for _, symbol_info in self._symbol_watcher._symbol_hooks.items():
            symbol_path = symbol_info['symbol']
            module_path = symbol_path.split(':')[0]
            
            logger.debug(f"Checking module {module_path} for symbol {symbol_path}")
            logger.debug(f"  - Module in sys.modules: {module_path in sys.modules}")
            logger.debug(f"  - Symbol already applied: {symbol_path in self._symbol_watcher._applied_hooks}")
            
            # 检查模块是否已导入，且该 symbol 尚未应用
            if module_path in sys.modules and symbol_path not in self._symbol_watcher._applied_hooks:
                logger.debug(f"Module {module_path} already loaded, applying hooks immediately")
                # 模拟模块加载完成事件
                self._symbol_watcher._on_symbol_module_loaded(module_path)
