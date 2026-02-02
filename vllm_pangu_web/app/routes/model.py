#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型API路由
"""

from fastapi import APIRouter,Request
from fastapi.responses import JSONResponse
from typing import Dict

from app.models.state import global_state
from app.models.schemas import HealthResponse

router = APIRouter(tags=["模型"])

@router.get("/health")
async def health_route():
    """健康检查"""
    return HealthResponse(
        status="ok",
        model_loaded=global_state.model_status["loaded"],
        model_name=global_state.model_status["model_name"],
        model_type=global_state.model_status["model_type"],
        device_type=global_state.model_status["device_type"],
        tensor_parallel_size=global_state.model_status["tensor_parallel_size"],
        quantization=global_state.model_status["quantization"]
    )

@router.post("/switch_model")
async def switch_model_route(request: Request):
    """切换模型"""
    from app.services.model_service import unload_model, load_model_with_config
    
    try:
        # 先卸载当前模型
        unload_result = await unload_model()
        if unload_result["status"] != "success":
            return JSONResponse(unload_result)
        
        # 加载新模型
        config = await request.json()
        load_result = await load_model_with_config(config)
        return JSONResponse(load_result)
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"模型切换失败: {str(e)}"})