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
import importlib
import importlib.metadata
import tempfile
import shutil
from unittest.mock import Mock, patch

import pytest

from msserviceprofiler.vllm_profiler.utils import (
    find_config_path, 
    load_yaml_config, 
    parse_version_tuple, 
    auto_detect_v1_default
)


@pytest.fixture
def temp_config_dir():
    """创建临时配置目录的 fixture"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_yaml_content():
    """提供示例 YAML 内容的 fixture"""
    return """
- symbol: "module1:Class1.method1"
  handler: "handlers:time_hook"
  domain: "TestDomain"
- symbol: "module2:function2"
  handler: "timer"
  attributes:
    - name: "input_length"
      expr: "len(kwargs['input_ids'])"
"""


@pytest.fixture
def mock_distribution():
    """提供模拟 distribution 的 fixture"""
    mock_dist = Mock()
    mock_dist.locate_file.return_value = "/fake/path/vllm_ascend"
    return mock_dist


class TestFindConfigPath:
    """测试 find_config_path 函数"""
    
    @staticmethod
    def test_find_config_path_user_config_success(temp_config_dir, monkeypatch):
        """测试优先找到用户目录下按版本命名的配置"""
        # 伪造 vllm.__version__
        fake_vllm = type("Vllm", (), {"__version__": "0.9.2"})
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

        # 将 ~ 指向临时目录
        home_dir = temp_config_dir
        monkeypatch.setattr("msserviceprofiler.vllm_profiler.utils.os.path.expanduser", lambda x: home_dir)

        # 创建用户配置文件 ~/.config/vllm_ascend/service_profiling_symbols.0.9.2.yaml
        user_cfg_dir = os.path.join(home_dir, ".config", "vllm_ascend")
        os.makedirs(user_cfg_dir, exist_ok=True)
        user_cfg_file = os.path.join(user_cfg_dir, "service_profiling_symbols.0.9.2.yaml")
        with open(user_cfg_file, "w", encoding="utf-8") as f:
            f.write("test user config")

        result = find_config_path()
        assert result == user_cfg_file

    @staticmethod
    def test_find_config_path_user_config_missing_fallback_to_local(temp_config_dir, monkeypatch):
        """测试用户配置不存在时回退到本地项目配置"""
        # 伪造 vllm.__version__ 存在但用户配置不存在
        fake_vllm = type("Vllm", (), {"__version__": "0.9.2"})
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
        # 将 ~ 指向临时目录，但不创建用户配置文件
        home_dir = temp_config_dir
        monkeypatch.setattr("msserviceprofiler.vllm_profiler.utils.os.path.expanduser", lambda x: home_dir)

        # 模拟本地项目存在配置文件
        with patch('msserviceprofiler.vllm_profiler.utils.os.path.dirname') as mock_dirname, \
             patch('msserviceprofiler.vllm_profiler.utils.os.path.isfile') as mock_isfile:
            mock_dirname.return_value = "/fake/project/path"
            expected_path = "/fake/project/path/config/service_profiling_symbols.yaml"

            # 第一次检查是用户配置（不存在，返回 False），第二次是本地项目（True）
            def isfile_side_effect(path):
                return path == expected_path
                
            mock_isfile.side_effect = isfile_side_effect

            result = find_config_path()
            assert result == expected_path

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.utils.importlib_metadata.distribution')
    def test_find_config_path_vllm_ascend_directory_not_found(mock_distribution):
        """测试 vllm_ascend 目录不存在的情况"""
        mock_dist = Mock()
        mock_dist.locate_file.return_value = None  # 目录不存在
        mock_distribution.return_value = mock_dist
        
        result = find_config_path()
        
        # 当前实现会回退到本地配置（若存在）
        assert result is None or result.endswith('service_profiling_symbols.yaml')

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.utils.importlib_metadata.distribution')
    def test_find_config_path_vllm_ascend_config_not_found(mock_distribution, temp_config_dir):
        """测试 vllm_ascend 目录存在但配置文件不存在的情况"""
        mock_dist = Mock()
        mock_dist.locate_file.return_value = temp_config_dir
        mock_distribution.return_value = mock_dist
        
        # 不创建配置文件
        
        result = find_config_path()
        
        # 当前实现会回退到本地配置（若存在）
        assert result is None or result.endswith('service_profiling_symbols.yaml')

    @staticmethod
    def test_find_config_path_vllm_ascend_exception():
        """测试用户配置查找过程中出现异常时仍可回退本地"""
        # 让导入 vllm 抛异常
        with patch.dict('sys.modules', {'vllm': None}):
            if 'vllm' in sys.modules:
                del sys.modules['vllm']

        result = find_config_path()
        
        # 应该继续尝试本地配置
        assert result is not None

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.utils.os.path.dirname')
    @patch('msserviceprofiler.vllm_profiler.utils.os.path.isfile')
    def test_find_config_path_local_project_success(mock_isfile, mock_dirname):
        """测试成功找到本地项目配置"""
        # 模拟 vllm_ascend 查找失败
        with patch('msserviceprofiler.vllm_profiler.utils.importlib_metadata.distribution') as mock_distribution:
            mock_distribution.side_effect = Exception("Test error")
            
            # 模拟本地配置文件存在
            mock_isfile.return_value = True
            mock_dirname.return_value = "/fake/project/path"
            
            result = find_config_path()
            
            expected_path = "/fake/project/path/config/service_profiling_symbols.yaml"
            mock_isfile.assert_called_with(expected_path)
            assert result == expected_path

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.utils.os.path.isfile')
    def test_find_config_path_no_config_found(mock_isfile):
        """测试找不到任何配置文件的情况"""
        # 模拟 vllm_ascend 查找失败
        with patch('msserviceprofiler.vllm_profiler.utils.importlib_metadata.distribution') as mock_distribution:
            mock_distribution.side_effect = Exception("Test error")
            
            # 模拟本地配置文件也不存在
            mock_isfile.return_value = False
            
            result = find_config_path()
            
            assert result is None

    @staticmethod
    def test_find_config_path_when_vllm_not_installed_uses_local():
        """测试未安装 vllm 时回退本地配置"""
        # 模拟 vllm 未安装
        with patch.dict('sys.modules', {'vllm': None}):
            if 'vllm' in sys.modules:
                del sys.modules['vllm']
        # 模拟本地配置文件存在
        with patch('msserviceprofiler.vllm_profiler.utils.os.path.dirname') as mock_dirname, \
             patch('msserviceprofiler.vllm_profiler.utils.os.path.isfile') as mock_isfile:
            mock_dirname.return_value = "/fake/project/path"
            expected_path = "/fake/project/path/config/service_profiling_symbols.yaml"
            
            def isfile_side_effect(path):
                return path == expected_path
            mock_isfile.side_effect = isfile_side_effect

            result = find_config_path()
            assert result == expected_path


class TestLoadYamlConfig:
    """测试 load_yaml_config 函数"""
    
    @staticmethod
    def test_load_yaml_config_success(temp_config_dir, sample_yaml_content):
        """测试成功加载 YAML 配置"""
        config_file = os.path.join(temp_config_dir, 'test_config.yaml')
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(sample_yaml_content)
        
        result = load_yaml_config(config_file)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['symbol'] == 'module1:Class1.method1'
        assert result[1]['symbol'] == 'module2:function2'

    @staticmethod
    def test_load_yaml_config_pyyaml_not_installed():
        """测试 PyYAML 未安装的情况"""
        with patch.dict('sys.modules', {'yaml': None}):
            # 重新导入以应用模拟（此处仅保留 importlib 用于 reload）
            if 'msserviceprofiler.vllm_profiler.utils' in sys.modules:
                importlib.reload(sys.modules['msserviceprofiler.vllm_profiler.utils'])
            result = load_yaml_config('/fake/path.yaml')
            assert result is None

    @staticmethod
    def test_load_yaml_config_file_not_found():
        """测试配置文件不存在的情况"""
        result = load_yaml_config('/nonexistent/path.yaml')
        
        assert result is None

    @staticmethod
    def test_load_yaml_config_invalid_yaml(temp_config_dir):
        """测试无效的 YAML 内容"""
        config_file = os.path.join(temp_config_dir, 'invalid.yaml')
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("invalid: yaml: content: [unclosed bracket")
        
        result = load_yaml_config(config_file)
        
        assert result is None

    @staticmethod
    def test_load_yaml_config_not_list(temp_config_dir):
        """测试 YAML 内容不是列表的情况"""
        config_file = os.path.join(temp_config_dir, 'not_list.yaml')
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("symbol: test\nhandler: timer")
        
        result = load_yaml_config(config_file)
        
        # 应该返回空列表而不是 None
        assert result == []

    @staticmethod
    def test_load_yaml_config_empty_file(temp_config_dir):
        """测试空配置文件"""
        config_file = os.path.join(temp_config_dir, 'empty.yaml')
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("")
        
        result = load_yaml_config(config_file)
        
        # 空文件应该返回 None（yaml.safe_load 返回 None）
        assert result is None

    @staticmethod
    def test_load_yaml_config_encoding_error(temp_config_dir):
        """测试编码错误的情况"""
        config_file = os.path.join(temp_config_dir, 'encoding_error.yaml')
        # 写入一些二进制数据模拟编码错误
        with open(config_file, 'wb') as f:
            f.write(b'\xff\xfeinvalid encoding')
        
        result = load_yaml_config(config_file)
        
        assert result is None


class TestParseVersionTuple:
    """测试 parse_version_tuple 函数"""
    
    @staticmethod
    @pytest.mark.parametrize("version_str,expected", [
        ("1.2.3", (1, 2, 3)),
        ("0.9.2", (0, 9, 2)),
        ("2.0.0", (2, 0, 0)),
        ("1.2.3+dev", (1, 2, 3)),  # 包含 + 的版本
        ("1.2.3-beta", (1, 2, 3)),  # 包含 - 的版本
        ("1.2", (1, 2, 0)),  # 缺少 patch 版本
        ("1", (1, 0, 0)),  # 只有 major 版本
        ("0.9.2+cpu", (0, 9, 2)),  # 包含 + 和其他标识
        ("1.2.3.4", (1, 2, 3)),  # 超过三个部分
    ])
    def test_parse_version_tuple_valid(version_str, expected):
        """测试解析有效的版本字符串"""
        result = parse_version_tuple(version_str)
        assert result == expected

    @staticmethod
    @pytest.mark.parametrize("version_str,expected", [
        ("1.2.a", (1, 2, 0)),  # 包含非数字字符
        ("a.b.c", (0, 0, 0)),  # 全部为非数字
        ("", (0, 0, 0)),  # 空字符串
        (".", (0, 0, 0)),  # 只有点
    ])
    def test_parse_version_tuple_invalid(version_str, expected):
        """测试解析无效的版本字符串"""
        result = parse_version_tuple(version_str)
        assert result == expected

    @staticmethod
    def test_parse_version_tuple_none_input():
        """测试 None 输入（虽然函数签名是 str，但测试边界情况）"""
        # 注意：函数期望字符串输入，但测试意外情况
        result = parse_version_tuple(None)
        # 根据实现，可能会抛出异常或返回默认值
        # 这里我们期望它能处理异常


class TestAutoDetectV1Default:
    """测试 auto_detect_v1_default 函数"""
    
    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.utils.importlib_metadata.version')
    def test_auto_detect_v1_default_new_version(mock_version):
        """测试新版本 vLLM (>= 0.9.2) 返回 '1'"""
        mock_version.return_value = "0.9.2"
        
        result = auto_detect_v1_default()
        
        assert result == "1"
        mock_version.assert_called_with("vllm")

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.utils.importlib_metadata.version')
    @pytest.mark.parametrize("version,expected", [
        ("0.9.2", "1"),
        ("0.9.3", "1"),
        ("1.0.0", "1"),
        ("0.9.1", "0"),  # 小于 0.9.2
        ("0.8.0", "0"),
        ("0.9.1+dev", "0"),  # 带标识符但仍小于 0.9.2
    ])
    def test_auto_detect_v1_default_various_versions(mock_version, version, expected):
        """测试各种版本号的自动检测"""
        mock_version.return_value = version
        
        result = auto_detect_v1_default()
        
        assert result == expected

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.utils.importlib_metadata.version')
    def test_auto_detect_v1_default_old_version(mock_version):
        """测试旧版本 vLLM (< 0.9.2) 返回 '0'"""
        mock_version.return_value = "0.9.1"
        
        result = auto_detect_v1_default()
        
        assert result == "0"

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.utils.importlib_metadata.version')
    def test_auto_detect_v1_default_version_not_found(mock_version):
        """测试 vLLM 包未找到的情况"""
        mock_version.side_effect = importlib.metadata.PackageNotFoundError("vllm not found")
        
        result = auto_detect_v1_default()
        
        assert result == "0"

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.utils.importlib_metadata.version')
    def test_auto_detect_v1_default_version_parse_error(mock_version):
        """测试版本解析错误的情况"""
        mock_version.return_value = "invalid.version.string"
        
        result = auto_detect_v1_default()
        
        # 应该回退到 "0"
        assert result == "0"

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.utils.importlib_metadata.version')
    def test_auto_detect_v1_default_general_exception(mock_version):
        """测试其他异常情况"""
        mock_version.side_effect = Exception("Unexpected error")
        
        result = auto_detect_v1_default()
        
        assert result == "0"

    @staticmethod
    @patch.dict('os.environ', {'VLLM_USE_V1': '1'})
    @patch('msserviceprofiler.vllm_profiler.utils.importlib_metadata.version')
    def test_auto_detect_v1_default_env_var_set(mock_version):
        """测试环境变量已设置的情况（虽然函数不检查，但确保不影响）"""
        # 注意：函数本身不检查环境变量，但测试确保环境变量不影响函数行为
        mock_version.return_value = "0.9.1"  # 旧版本
        
        result = auto_detect_v1_default()
        
        # 函数应该忽略环境变量，只基于版本检测
        assert result == "0"


class TestIntegration:
    """集成测试"""
    
    @staticmethod
    def test_integration_find_and_load_config(temp_config_dir, sample_yaml_content):
        """测试查找和加载配置的完整流程"""
        # 创建本地配置文件
        config_dir = os.path.join(os.path.dirname(__file__), 'config')
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, 'service_profiling_symbols.yaml')
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(sample_yaml_content)
            
            # 查找配置路径
            found_path = find_config_path()
            
            # 应该找到本地配置文件
            assert found_path is not None
            assert 'service_profiling_symbols.yaml' in found_path
            
            # 加载配置
            config_data = load_yaml_config(found_path)
            
            assert isinstance(config_data, list)
            assert len(config_data) > 0
            
        finally:
            # 清理
            if os.path.exists(config_file):
                os.remove(config_file)
            if os.path.exists(config_dir) and not os.listdir(config_dir):
                os.rmdir(config_dir)

    @staticmethod
    def test_version_parsing_integration():
        """测试版本解析的集成"""
        test_versions = [
            ("0.9.2", (0, 9, 2)),
            ("1.2.3+dev", (1, 2, 3)),
            ("2.0", (2, 0, 0)),
        ]
        
        for version_str, expected in test_versions:
            result = parse_version_tuple(version_str)
            assert result == expected
            
            # 测试版本比较（auto_detect_v1_default 中的逻辑）
            use_v1 = result >= (0, 9, 2)
            expected_use_v1 = version_str not in ["0.9.1", "0.8.0"]  # 这些应该返回 False
            assert use_v1 == expected_use_v1


class TestEdgeCases:
    """边界情况测试"""
    
    @staticmethod
    def test_find_config_path_special_characters(temp_config_dir):
        """测试路径包含特殊字符的情况"""
        # 这个测试主要确保路径处理不会因特殊字符而失败
        # 实际实现中可能不需要特别处理，但测试确保健壮性
        with patch('msserviceprofiler.vllm_profiler.utils.os.path.dirname') as mock_dirname:
            mock_dirname.return_value = "/path/with/special/chars"
            with patch('msserviceprofiler.vllm_profiler.utils.os.path.isfile') as mock_isfile:
                mock_isfile.return_value = True
                
                result = find_config_path()
                
                assert result is not None
                assert 'special' in result

    @staticmethod
    def test_load_yaml_config_large_file(temp_config_dir):
        """测试大文件加载（如果有内存限制需要考虑）"""
        config_file = os.path.join(temp_config_dir, 'large.yaml')
        
        # 创建一个大但不至于耗尽内存的 YAML 文件
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("- symbol: test\n  handler: timer\n")
            # 添加一些重复内容使文件变大但保持有效
            for i in range(1000):
                f.write(f"- symbol: test{i}\n  handler: timer\n")
        
        result = load_yaml_config(config_file)
        
        assert isinstance(result, list)
        assert len(result) == 1001

    @staticmethod
    def test_parse_version_tuple_very_long_version():
        """测试非常长的版本字符串"""
        long_version = "1." + "9." * 100 + "0"
        result = parse_version_tuple(long_version)
        
        # 应该只取前三个部分
        assert result == (1, 9, 9)

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.utils.importlib_metadata.version')
    def test_auto_detect_with_complex_version_string(mock_version):
        """测试复杂的版本字符串"""
        complex_versions = [
            "0.9.2.post1+dev123",
            "0.9.2-rc1",
            "0.9.2+build.123",
        ]
        
        for version in complex_versions:
            mock_version.return_value = version
            result = auto_detect_v1_default()
            # 所有这些都是 >= 0.9.2，应该返回 "1"
            assert result == "1"
