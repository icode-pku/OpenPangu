#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局状态管理
"""

from typing import Dict, List, Optional, Any
from vllm.engine.async_llm_engine import AsyncLLMEngine

class GlobalState:
    """全局状态管理类"""
    
    def __init__(self):
        # 模型引擎
        self.engine: Optional[AsyncLLMEngine] = None
        self.ascend_session: Optional[Any] = None
        
        # 当前模型配置
        self.current_model_config: Dict[str, Any] = {}
        
        # 对话历史
        self.conversation_histories: Dict[str, List[Dict]] = {}
        
        # 模型状态
        self.model_status: Dict[str, Any] = {
            "loaded": False,
            "loading": False,
            "model_name": "",
            "model_type": "",  # "ascend" 或 "vllm"
            "device_type": "cpu",
            "quantization": "none",
            "load_time": None,
            "memory_used": 0,
            "tensor_parallel_size": 1
        }
        
        # 硬件可用性
        self.ascend_available: bool = False
        self.vllm_available: bool = False
        self.quantization_available: bool = False
    
    def clear_conversations(self):
        """清空所有对话历史"""
        self.conversation_histories.clear()
    
    def get_conversation(self, session_id: str) -> List[Dict]:
        """获取对话历史"""
        return self.conversation_histories.get(session_id, [])
    
    def update_conversation(self, session_id: str, history: List[Dict]):
        """更新对话历史"""
        self.conversation_histories[session_id] = history

# 全局状态实例
global_state = GlobalState()