#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI应用创建
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.models.state import global_state
from app.utils.hardware import detect_hardware
from app.routes import admin, conversation, model
from config import TEMPLATES_DIR, STATIC_DIR_1, STATIC_DIR_2

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时清理历史
    global_state.clear_conversations()
    print("🧹 已清理所有之前的会话历史")
    
    # 检测硬件环境
    hardware_info = detect_hardware()
    global_state.vllm_available = hardware_info["vllm_available"] if hardware_info["vllm_available"] else True
    global_state.quantization_available = hardware_info["quantization_available"]
    global_state.model_status["device_type"] = "npu" if hardware_info["npu_smi_info"] else "cpu"
    
    print("ℹ️  模型未自动加载，请通过web页面加载模型")
    
    yield
    
    # 关闭时卸载模型
    from app.services.model_service import unload_model
    await unload_model()
    print("👋 服务关闭完成")

def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="盘古大模型Web接口(昇腾NPU版)", 
        description="基于openPangu-Embedded的智能对话系统，支持昇腾NPU推理",
        lifespan=lifespan
    )
    
    # 初始化模板
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
    app.state.templates = templates
    
    # 检查模板文件
    index_path = os.path.join("templates", "index.html")
    
    if os.path.exists(index_path):
        print(f"✅ 找到前端页面: {index_path}")
    else:
        print(f"❌ 未找到前端页面: {index_path}")

    # 挂载静态文件
    if os.path.exists(STATIC_DIR_1):     
        app.mount("/templates/css", StaticFiles(directory=STATIC_DIR_1), name="static")
    if os.path.exists(STATIC_DIR_2):
        app.mount("/templates/js", StaticFiles(directory=STATIC_DIR_2), name="static")
    
    # 注册路由
    app.include_router(admin.router)
    app.include_router(conversation.router)
    app.include_router(model.router)

    # 调试：打印已注册的路由
    print("\n🔍 已注册的路由列表:")
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            methods = ", ".join(route.methods)
            path = route.path
            print(f"  {methods:15} {path}")
    
    return app