#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提示词构建模块
"""

from typing import List, Dict

def build_context_prompt(history: List[Dict], current_prompt: str) -> str:
    """构建包含对话历史的prompt"""
    prompt_parts = []
    
    # 系统指令
    prompt_parts.append("重要指令：")
    prompt_parts.append("1. 直接回答问题，不要重复用户的问题")
    prompt_parts.append("2. 保持回答简洁明了，避免冗长")
    prompt_parts.append("3. 如果不确定答案，可以说明不确定，而不是编造信息")
    prompt_parts.append("")
    
    # 过滤完整的对话轮次
    completed_history = []
    i = 0
    while i < len(history):
        if i + 1 < len(history) and history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
            completed_history.append(history[i])
            completed_history.append(history[i+1])
            i += 2
        else:
            i += 1
    
    # 取最近的6轮对话
    recent_history = completed_history[-6:]
    
    # 构建对话历史
    for turn in recent_history:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            prompt_parts.append(f"用户: {content}")
        elif role == "assistant":
            prompt_parts.append(f"助手: {content}")
    
    # 添加当前问题
    prompt_parts.append(f"用户: {current_prompt}")
    prompt_parts.append("助手: ")
    
    return "\n".join(prompt_parts)

def update_conversation_history(history: List[Dict], user_prompt: str, assistant_response: str, max_history: int = 20) -> List[Dict]:
    """更新对话历史"""
    new_history = history.copy()
    new_history.append({"role": "user", "content": user_prompt})
    new_history.append({"role": "assistant", "content": assistant_response})
    
    # 限制历史长度
    if len(new_history) > max_history:
        new_history = new_history[-max_history:]
    
    return new_history

def filter_unused_content(current_text: str) -> str:
    """去除 [unused17] 前面的所有文本"""
    if "[unused17]" not in current_text:
        return current_text

    unused17_index = current_text.find("[unused17]")
    return current_text[unused17_index + len("[unused17]"):]