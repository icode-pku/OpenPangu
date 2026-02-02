#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理服务
"""

import time
from typing import Dict, Optional, Any
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext

from app.models.state import global_state
from config import MODEL_PATH, TENSOR_PARALLEL_SIZE

async def load_vllm_model(config: Dict) -> Dict[str, Any]:
    """加载vLLM模型"""
    try:
        model_path = config.get("model_path", MODEL_PATH)
        quantization = config.get("quantization", "none")
        
        print(f"🚀 开始加载vLLM模型: {model_path}")
        print(f"🔢 量化方法: {quantization}")
    
        # 使用配置的张量并行大小
        tensor_parallel_size = config.get("tensor_parallel_size", TENSOR_PARALLEL_SIZE)
        
        # 构建引擎参数
        engine_args_dict = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "max_num_seqs": config.get("max_num_seqs", 32),
            "max_model_len": config.get("max_model_len", 16384),
            "max_num_batched_tokens": config.get("max_num_batched_tokens", 4096),
            "tokenizer_mode": "slow",
            "gpu_memory_utilization": config.get("gpu_memory_utilization", 0.9),
            "dtype": "auto",
        }
        
        print(f"🔧 使用张量并行大小: {tensor_parallel_size}")
        print(f"🔧 量化配置: {quantization}")
        
        engine_args = AsyncEngineArgs(**engine_args_dict)
        
        # 加载引擎
        start_time = time.time()
        engine = AsyncLLMEngine.from_engine_args(
            engine_args,
            usage_context=UsageContext.API_SERVER
        )
        load_time = time.time() - start_time
        
        # 更新全局状态
        global_state.engine = engine
        global_state.current_model_config = config.copy()
        global_state.model_status.update({
            "loaded": True,
            "loading": False,
            "model_name": model_path.split("/")[-1],
            "model_type": "vllm",
            "quantization": quantization,
            "load_time": load_time,
            "memory_used": config.get("gpu_memory_utilization", 0.9) * 100,
            "tensor_parallel_size": tensor_parallel_size
        })
        
        print(f"✅ vLLM模型加载成功")
        print(f"⏱️  加载时间: {load_time:.2f}秒")

        # ===== 安全的模型信息打印 =====
        print("\n" + "="*50)
        print("📋 已加载的模型结构信息:")
        print("="*50)
        
        # 获取模型配置
        model_config = engine.model_config

        if hasattr(model_config, 'model'):
            print(f"模型路径: {model_config.model}")
        
        if hasattr(model_config, 'architecture'):
            print(f"模型架构: {model_config.architecture}")
        
        if hasattr(model_config, 'hidden_size'):
            print(f"隐藏层维度: {model_config.hidden_size}")
        
        if hasattr(model_config, 'hf_config'):
            print(f"HuggingFace配置对象: {model_config.hf_config}")
        
        if hasattr(model_config, 'dtype'):
            print(f"数据类型: {model_config.dtype}")
        
        return {
            "status": "success", 
            "message": "vLLM模型加载成功",
            "load_time": load_time,
            "model_name": global_state.model_status["model_name"],
            "device_type": "gpu",
            "tensor_parallel_size": tensor_parallel_size,
            "quantization": quantization
        }
        
    except Exception as e:
        global_state.model_status["loading"] = False
        global_state.model_status["loaded"] = False
        print(f"❌ vLLM模型加载失败: {e}")
        return {"status": "error", "message": f"vLLM模型加载失败: {str(e)}"}

async def load_model_with_config(config: Dict) -> Dict[str, Any]:
    """根据配置加载模型"""
    if global_state.model_status["loading"]:
        return {"status": "error", "message": "模型正在加载中"}
    
    if global_state.model_status["loaded"]:
        return {"status": "error", "message": "模型已加载，请先卸载"}
    
    global_state.model_status["loading"] = True
    
    model_type = config.get("model_type", "vllm")
    
    if model_type == "vllm" and global_state.vllm_available:
        return await load_vllm_model(config)
    else:
        if global_state.vllm_available:
            return await load_vllm_model(config)
        else:
            global_state.model_status["loading"] = False
            return {"status": "error", "message": "无可用推理后端"}

async def unload_model() -> Dict[str, Any]:
    """卸载模型"""
    if not global_state.model_status["loaded"]:
        return {"status": "warning", "message": "没有加载的模型"}
    
    try:
        print("🔄 开始卸载模型...")
        
        if global_state.model_status["model_type"] == "vllm" and global_state.engine is not None:
            # 清理vLLM引擎
            try:
                global_state.engine.shutdown()
            except:
                pass
            global_state.engine = None
            print("✅ vLLM模型卸载成功")
        
        # 重置状态
        global_state.model_status.update({
            "loaded": False,
            "loading": False,
            "model_name": "",
            "model_type": "",
            "quantization": "none",
            "load_time": None,
            "memory_used": 0,
            "tensor_parallel_size": 1
        })
        global_state.current_model_config = {}
        
        # 清理对话历史
        global_state.clear_conversations()
        print("🧹 已清理所有对话历史")
        
        return {"status": "success", "message": "模型卸载成功"}
        
    except Exception as e:
        print(f"❌ 模型卸载失败: {e}")
        return {"status": "error", "message": f"模型卸载失败: {str(e)}"}