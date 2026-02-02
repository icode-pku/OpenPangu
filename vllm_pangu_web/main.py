#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主程序入口
"""

import os
import uvicorn
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.app import create_app
from config import HOST, PORT, LOG_LEVEL

def main():
    """主函数"""
    
    # 创建应用
    app = create_app()
    
    # 添加根路由
    @app.get("/")
    async def read_root(request: Request):
        from fastapi.templating import Jinja2Templates
        templates = Jinja2Templates(directory="templates")
        return templates.TemplateResponse("index.html", {"request": request})
    
    # 打印启动信息
    print_startup_info()
    
    # 启动服务器
    uvicorn.run(
        app, 
        host=HOST, 
        port=PORT, 
        log_level=LOG_LEVEL,
        timeout_keep_alive=300
    )

def print_startup_info():
    """打印启动信息"""
    
    print(f"🚀 启动盘古大模型对话推理与性能测试Web服务(昇腾NPU版)...")
    print(f"📡 服务地址: http://{HOST}:{PORT}")
    print(f"🌐 网页界面: http://localhost:{PORT}")

if __name__ == "__main__":
    main()