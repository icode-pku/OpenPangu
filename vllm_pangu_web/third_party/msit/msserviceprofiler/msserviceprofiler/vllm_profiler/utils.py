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
import importlib.metadata as importlib_metadata
from typing import Optional, Dict, Any, List
from .logger import logger


def find_config_path() -> Optional[str]:
    """查找性能分析配置文件，按优先级顺序查找。
    
    查找顺序：
    1. 用户配置目录: ~/.config/vllm_ascend/service_profiling_symbols.{VLLM_VERSION}.yaml
    2. 本项目目录: <this>/config/service_profiling_symbols.yaml
    
    Returns:
        Optional[str]: 配置文件路径，如果未找到则返回 None
    """
    # 1) user config path: ~/.config/vllm_ascend/service_profiling_symbols.{VLLM_VERSION}.yaml
    try:
        try:
            import vllm  # type: ignore
            vllm_version = getattr(vllm, '__version__', None)
        except Exception as e:
            logger.debug(f"vllm not available for version detection: {e}")
            vllm_version = None
        
        try:
            from vllm_ascend import register_service_profiling # type: ignore
            register_service_profiling()
        except Exception as e:
            logger.debug(f"Cannot using register_service_profiling to get default symbols config: {e}")

        if vllm_version:
            home_dir = os.path.expanduser('~')
            candidate = os.path.join(
                home_dir,
                '.config',
                'vllm_ascend',
                f"service_profiling_symbols.{vllm_version}.yaml",
            )
            if os.path.isfile(candidate):
                logger.debug(f"Using profiling symbols from user config: {candidate}")
                return candidate
    except Exception as e:
        logger.warning(f"Failed to find profiling symbols from default path: {e}")

    # 2) local project config path
    local_candidate = os.path.join(os.path.dirname(__file__), 'config', 'service_profiling_symbols.yaml')
    if os.path.isfile(local_candidate):
        logger.debug(f"Using profiling symbols from local project: {local_candidate}")
        return local_candidate

    return None


def load_yaml_config(config_path: str) -> Optional[List[Dict[str, Any]]]:
    """加载 YAML 配置文件。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Optional[List[Dict[str, Any]]]: 配置数据列表，失败时返回 None
        
    Raises:
        ImportError: 当 PyYAML 未安装时
        FileNotFoundError: 当配置文件不存在时
    """
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML is required for configuration loading")
        return None
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config is None:
                return None
            if isinstance(config, list):
                return config
            logger.warning("Configuration file should be a list of hook configurations")
            return []
    except FileNotFoundError:
        logger.warning(f"Configuration file does not exist: {config_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load YAML configuration: {e}")
        return None


def parse_version_tuple(version_str: str) -> tuple:
    """解析版本字符串为元组。
    
    将版本字符串解析为 (major, minor, patch) 格式的元组。
    处理包含 "+" 或 "-" 的版本字符串，只取主要版本号部分。
    
    Args:
        version_str: 版本字符串，如 "1.2.3+dev" 或 "0.9.2"
        
    Returns:
        tuple: (major, minor, patch) 版本元组
        
    Example:
        >>> parse_version_tuple("1.2.3+dev")
        (1, 2, 3)
        >>> parse_version_tuple("0.9")
        (0, 9, 0)
    """
    if not isinstance(version_str, str):
        return (0, 0, 0)
    parts = version_str.split("+")[0].split("-")[0].split(".")
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            break
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])


def auto_detect_v1_default() -> str:
    """根据已安装的 vLLM 版本自动决定默认 V1 使用情况。
    
    启发式规则：对于较新的 vLLM (>= 0.9.2) 默认使用 V1，否则使用 V0。
    如果无法确定版本，为了安全起见回退到 V0。
    
    Returns:
        str: "1" 表示使用 V1，"0" 表示使用 V0
        
    Note:
        该方法会检查环境变量 VLLM_USE_V1，如果未设置则自动检测。
    """
    try:
        vllm_version = importlib_metadata.version("vllm")
        major, minor, patch = parse_version_tuple(vllm_version)
        use_v1 = (major, minor, patch) >= (0, 9, 2)
        logger.info(
            f"VLLM_USE_V1 not set, auto-detected via vLLM {vllm_version}: default {'1' if use_v1 else '0'}"
        )
        return "1" if use_v1 else "0"
    except Exception as e:
        logger.info("VLLM_USE_V1 not set and vLLM version unknown; default to 0 (V0)")
        return "0"
