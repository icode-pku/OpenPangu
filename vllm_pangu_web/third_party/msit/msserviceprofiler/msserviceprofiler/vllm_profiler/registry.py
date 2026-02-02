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

# 全局注册表存储所有hookers
HOOK_REGISTRY = []
# 仅用于在基于配置的场景下，临时保存已导入的内置 Hooker 列表
BUILTIN_HOOKERS_CACHE = []


def clear_hook_registry():
    """清空全局 hook 注册表。"""
    HOOK_REGISTRY.clear()


def clear_builtin_cache():
    """清空内置 hooker 缓存。"""
    BUILTIN_HOOKERS_CACHE.clear()


def get_hook_registry():
    """获取全局 hook 注册表。
    
    Returns:
        List: hook 注册表列表
    """
    return HOOK_REGISTRY


def get_builtin_cache():
    """获取内置 hooker 缓存。
    
    Returns:
        List: 内置 hooker 缓存列表
    """
    return BUILTIN_HOOKERS_CACHE


def add_to_hook_registry(hooker):
    """添加 hooker 到全局注册表。
    
    Args:
        hooker: 要添加的 hooker 实例
    """
    HOOK_REGISTRY.append(hooker)


def set_builtin_cache(cache):
    """设置内置 hooker 缓存。
    
    Args:
        cache: 要设置的缓存列表
    """
    if cache is None:
        BUILTIN_HOOKERS_CACHE.clear()
        return
    # 就地替换，保持引用不变
    BUILTIN_HOOKERS_CACHE.clear()
    BUILTIN_HOOKERS_CACHE.extend(list(cache))
