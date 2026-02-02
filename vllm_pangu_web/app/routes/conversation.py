#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话API路由
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict

from app.models.schemas import GenerateRequest
from app.models.state import global_state
from app.services.conversation_service import (
    start_conversation,
    clear_conversation,
    get_conversation_history,
    generate_vllm_response
)

router = APIRouter(prefix="/conversation", tags=["对话"])

@router.post("/start")
async def start_conversation_route():
    """开始新的对话"""
    result = start_conversation()
    return JSONResponse(result)

@router.post("/generate")
async def generate_route(request: Request):
    """生成对话"""
    if not global_state.model_status["loaded"]:
        return JSONResponse({"error": "模型未加载，请先加载模型"}, status_code=400)
    
    request_dict = await request.json()
    prompt = request_dict.get("prompt", "")
    session_id = request_dict.get("session_id")
    stream = request_dict.get("stream", False)
    
    # 如果没有session_id，创建新的对话
    if not session_id:
        result = start_conversation()
        session_id = result["session_id"]
    
    # 准备采样参数
    sampling_params = {
        "max_tokens": request_dict.get("max_tokens", 2048),
        "temperature": request_dict.get("temperature", 0.7),
        "top_p": request_dict.get("top_p", 0.9),
        "repetition_penalty": request_dict.get("repetition_penalty", 1.1),
    }
    
    # 根据模型类型选择推理方式
    if global_state.model_status["model_type"] == "vllm":
        if stream:
            return StreamingResponse(
                generate_vllm_response(prompt, session_id, sampling_params),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # 非流式响应（简化实现）
            return JSONResponse({
                "text": ["非流式响应功能需要完整实现"],
                "session_id": session_id,
                "model_type": "vllm"
            })
    else:
        return JSONResponse({"error": "未知的模型类型"}, status_code=500)

@router.post("/{session_id}/clear")
async def clear_conversation_route(session_id: str):
    """清空指定对话的历史"""
    result = clear_conversation(session_id)
    if "error" in result:
        return JSONResponse(result, status_code=404)
    return JSONResponse(result)

@router.get("/{session_id}/history")
async def get_conversation_history_route(session_id: str):
    """获取对话历史"""
    result = get_conversation_history(session_id)
    return JSONResponse(result)