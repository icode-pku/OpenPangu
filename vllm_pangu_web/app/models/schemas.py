#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pydantic数据模型
"""

from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field

class ModelLoadRequest(BaseModel):
    """模型加载请求"""
    preset: Optional[str] = None
    model_type: Optional[str] = "vllm"
    model_path: Optional[str] = None
    tensor_parallel_size: Optional[int] = 4
    quantization: Optional[str] = "none"
    gpu_memory_utilization: Optional[float] = 0.9

class QuantizationRequest(BaseModel):
    """量化请求"""
    model_path: str
    output_path: str
    quantization_bits: int = Field(4, ge=4, le=8)

class GenerateRequest(BaseModel):
    """生成请求"""
    prompt: str
    session_id: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stream: bool = False

class ConversationRequest(BaseModel):
    """对话请求"""
    prompt: str
    session_id: Optional[str] = None

class ModelStatusResponse(BaseModel):
    """模型状态响应"""
    loaded: bool
    loading: bool
    model_name: str
    model_type: str
    device_type: str
    quantization: str
    load_time: Optional[float]
    memory_used: float
    tensor_parallel_size: int

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    model_name: str
    model_type: str
    device_type: str
    tensor_parallel_size: int
    quantization: str

class SystemInfoResponse(BaseModel):
    """系统信息响应"""
    cpu_count: int
    cpu_percent: float
    memory_total: float
    memory_used: float
    memory_percent: float
    disk_total: float
    disk_used: float
    disk_percent: float