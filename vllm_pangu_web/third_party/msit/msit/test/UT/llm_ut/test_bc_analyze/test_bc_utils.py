import pytest
import os
from unittest.mock import patch, MagicMock
import random
from msit_llm.bc_analyze import RandomNameSequence

class TestRandomNameSequence:
    
    def test_character_set(self):
        assert set(RandomNameSequence.characters) == set("abcdefghijklmnopqrstuvwxyz0123456789_")
    
    def test_name_generation(self):
        """测试生成随机名称的基本功能"""
        rns = RandomNameSequence()
        names = [next(rns) for _ in range(10)]
        
        for name in names:
            assert len(name) == 8
            assert all(c in RandomNameSequence.characters for c in name)