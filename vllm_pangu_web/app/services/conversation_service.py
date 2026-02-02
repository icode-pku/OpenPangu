#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话服务模块
"""

import uuid
import asyncio
import json
from typing import Dict, List, AsyncGenerator
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from app.models.state import global_state
from app.utils.prompt_builder import (
    build_context_prompt, 
    update_conversation_history,
    filter_unused_content
)

def start_conversation() -> Dict[str, str]:
    """开始新的对话"""
    session_id = str(uuid.uuid4())
    global_state.conversation_histories[session_id] = []
    return {"session_id": session_id, "status": "started"}

def clear_conversation(session_id: str) -> Dict[str, str]:
    """清空指定对话的历史"""
    if session_id in global_state.conversation_histories:
        global_state.conversation_histories[session_id] = []
        return {"status": "cleared"}
    return {"error": "Session not found"}

def get_conversation_history(session_id: str) -> Dict[str, List]:
    """获取对话历史"""
    history = global_state.conversation_histories.get(session_id, [])
    return {"history": history}

async def generate_vllm_response(
    prompt: str,
    session_id: str,
    sampling_params: Dict[str, any]
) -> AsyncGenerator[bytes, None]:
    """使用vLLM生成响应（流式）"""
    
    # 获取对话历史
    history = global_state.get_conversation(session_id)
    context_prompt = build_context_prompt(history, prompt)
    
    # 创建采样参数
    vllm_sampling_params = SamplingParams(**sampling_params)
    request_id = random_uuid()
    
    if global_state.engine is None:
        yield (json.dumps({"error": "Engine not initialized"}, ensure_ascii=False) + "\n").encode("utf-8")
        return
    
    generator = global_state.engine.generate(context_prompt, vllm_sampling_params, request_id)
    is_thinking_message = True
    previous_texts = {}
    finished_outputs = set()
    full_response = ""
    
    try:
        async for output in generator:
            all_finished = True
            
            for i, single_output in enumerate(output.outputs):
                current_text = single_output.text
                
                if i in finished_outputs:
                    continue
                
                if "[unused16]" in current_text: # 思考标记
                    if is_thinking_message:
                        if "[unused17]" in current_text:
                            current_text = filter_unused_content(current_text)
                            is_thinking_message = False
                        else:
                            current_text = ""
                    else:
                        current_text = filter_unused_content(current_text)
                else:
                    current_text = filter_unused_content(current_text)

                full_response = current_text
                
                if i not in previous_texts:
                    previous_texts[i] = ""
                    if current_text:
                        response_data = {
                            "text": current_text,
                            "index": i,
                            "finished": False
                        }
                        yield (json.dumps(response_data, ensure_ascii=False) + "\n").encode("utf-8")
                        await asyncio.sleep(0)
                else:
                    previous_text = previous_texts[i]
                    if current_text.startswith(previous_text):
                        new_text = current_text[len(previous_text):]
                        if new_text:
                            response_data = {
                                "text": new_text,
                                "index": i,
                                "finished": False
                            }
                            yield (json.dumps(response_data, ensure_ascii=False) + "\n").encode("utf-8")
                            await asyncio.sleep(0)
                    else:
                        if current_text:
                            response_data = {
                                "text": current_text,
                                "index": i,
                                "finished": False
                            }
                            yield (json.dumps(response_data, ensure_ascii=False) + "\n").encode("utf-8")
                            await asyncio.sleep(0)
                
                previous_texts[i] = current_text
                
                if (single_output.finish_reason is not None and 
                    i not in finished_outputs):
                    finished_outputs.add(i)
                    
                    # 更新对话历史
                    if full_response:
                        new_history = update_conversation_history(
                            history, prompt, full_response
                        )
                        global_state.update_conversation(session_id, new_history)
                    
                    finish_data = {
                        "text": "",
                        "index": i,
                        "finish_reason": single_output.finish_reason,
                        "finished": True,
                        "session_id": session_id
                    }
                    yield (json.dumps(finish_data, ensure_ascii=False) + "\n").encode("utf-8")
                    await asyncio.sleep(0)
                
                if single_output.finish_reason is None:
                    all_finished = False
            
            if all_finished and len(output.outputs) > 0:
                break
                    
    except Exception as e:
        print(f"流式输出错误: {e}")
        error_data = {
            "error": "生成中断",
            "finished": True
        }
        yield (json.dumps(error_data, ensure_ascii=False) + "\n").encode("utf-8")