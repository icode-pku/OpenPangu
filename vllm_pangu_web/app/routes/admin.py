#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
管理API路由
"""
import json  # 确保导入json模块
import os
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.models.state import global_state
from app.services.model_service import load_model_with_config, unload_model
from app.services.quantization_service import quantize_model_with_w8a8
from config import MODEL_PRESETS, check_quantize_model_exists

router = APIRouter(prefix="/admin", tags=["管理"])

@router.get("/model_status")
async def get_model_status_route():
    """获取模型状态"""
    return JSONResponse(global_state.model_status)

@router.post("/load_model")
async def load_model(request: Request):
    """加载模型"""
    try:
        config = await request.json()
        result = await load_model_with_config(config)
        return JSONResponse(result)
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"请求解析错误: {str(e)}"})

@router.post("/unload_model")
async def unload_model_route():
    """卸载模型"""
    result = await unload_model()
    return JSONResponse(result)

@router.post("/quantize_model")
async def quantize_model(request: Request):
    """量化模型API"""
    try:
        config = await request.json()
        
        model_path = config.get("model_path")
        output_path = config.get("quantized_path")
        quantization_bits = config.get("quantization_bits")
        max_sequence_length = config.get("max_sequence_length", 4096)
        num_calibration_samples = config.get("num_calibration_samples", 512)
        
        result = quantize_model_with_w8a8(model_path, output_path, quantization_bits, max_sequence_length, num_calibration_samples)
        return JSONResponse(result)
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"量化请求解析错误: {str(e)}"})



@router.post("/check_quantization_path")
async def check_quantization_path(request: Request):
    """检查量化模型是否存在指定路径"""
    config = await request.json()
    quantized_path = config.get("quantized_path")
    return JSONResponse({
        "quantization_available": global_state.quantization_available,
        "bnb_model_exists": check_quantize_model_exists(quantized_path),
        "quantized_path": quantized_path,
    })

@router.get("/available_presets")
async def get_available_presets_route():
    """获取可用的模型预设"""
    return JSONResponse({
        "presets": MODEL_PRESETS,
        "vllm_available": global_state.vllm_available,
        "quantization_available": global_state.quantization_available,
        "default_tensor_parallel_size": "配置中的并行大小",
        "bnb_model_exists": check_quantize_model_exists()
    })

@router.post("/clear_all_sessions")
async def clear_all_sessions_route():
    """清空所有会话"""
    session_count = len(global_state.conversation_histories)
    global_state.clear_conversations()
    
    return JSONResponse({
        "status": "success",
        "message": f"已清空 {session_count} 个会话",
        "cleared_sessions": session_count
    })

@router.post("/save_config")
async def save_model_config(request: Request):
    """保存模型配置"""
    try:
        config = await request.json()
        # 保存配置到文件或数据库
        with open("model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        return JSONResponse({"status": "success", "message": "配置保存成功"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

@router.get("/model_config")
async def get_model_config():
    """获取保存的模型配置"""
    try:
        if os.path.exists("model_config.json"):
            with open("model_config.json", "r") as f:
                config = json.load(f)
            return JSONResponse({"status": "success", "config": config})
        else:
            return JSONResponse({"status": "success", "config": {}})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

@router.get("/performance_metrics")
async def get_performance_metrics():
    """获取性能指标"""
    try:
        # 这里可以返回后端的性能指标
        metrics = {
            "ttft": global_state.metrics.get("ttft", 0),
            "itl": global_state.metrics.get("itl", 0),
            "tps": global_state.metrics.get("tps", 0),
            "e2e_latency": global_state.metrics.get("e2e_latency", 0),
            "rps": global_state.metrics.get("rps", 0),
            "total_requests": global_state.metrics.get("total_requests", 0),
            "total_tokens": global_state.metrics.get("total_tokens", 0)
        }
        return JSONResponse({"status": "success", "metrics": metrics})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})
    

