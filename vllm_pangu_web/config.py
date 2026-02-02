#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# 模型路径配置
MODEL_PATH = "/opt/pangu/openPangu-Embedded-7B-V1.1" 
W8A8_MODEL_PATH = "/opt/pangu/openPangu-Embedded-7B-V1.1-W8A8-Dynamic-Per-Token"

# NPU配置
NPU_DEVICE_ID = 0
NPU_MEMORY_UTILIZATION = 0.8

# vLLM配置
TENSOR_PARALLEL_SIZE = 4
MAX_NUM_BATCHED_TOKENS = 4096
MAX_NUM_SEQS = 32
MAX_MODEL_LEN = 16384

# 服务器配置
HOST = "0.0.0.0"
PORT = 1040
LOG_LEVEL = "info"

# 量化方法配置
QUANTIZATION_METHODS = {
    "none": None,
    "bnb_4bit": "bitsandbytes",
    # 可以根据需要添加更多量化方法
}

# 模型预设配置
MODEL_PRESETS = {
    "7b_vllm_full": {
        "model_type": "vllm",
        "model_path": MODEL_PATH,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": 0.9,
        "quantization": None,
        "description": f"7B模型 - vLLM全精度 (TP={TENSOR_PARALLEL_SIZE})"
    },
}

# 对话配置
MAX_HISTORY_LENGTH = 20
MAX_CONTEXT_LENGTH = 8192

# 文件路径配置
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR_1 = BASE_DIR / "templates" / "css"
STATIC_DIR_2 = BASE_DIR / "templates" / "js"


# 确保目录存在
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR_1.mkdir(exist_ok=True)
STATIC_DIR_2.mkdir(exist_ok=True)

def get_model_preset(preset_name: str) -> Optional[Dict]:
    """获取模型预设配置"""
    return MODEL_PRESETS.get(preset_name)

def check_quantize_model_exists(quantized_path: str) -> bool:
    """检查量化模型是否存在"""
    if not os.path.exists(quantized_path):
        return False
    
    # 检查必要的文件是否存在
    required_files = ["config.json", "tokenizer.model"]
    for file in required_files:
        if not os.path.exists(os.path.join(quantized_path, file)):
            return False
    
    return True