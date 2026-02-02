#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件检测模块
"""

import torch
import transformers
import subprocess
import re
import os
from typing import Dict, Any, Optional

def detect_hardware() -> Dict[str, bool]:
    """检测硬件环境"""
    hardware_info = {
        "ascend_available": False,
        "vllm_available": False,
        "quantization_available": False,
        "cuda_available": False,
        "npu_smi_info": None  # 新增：npu-smi信息
    }
    
    try:
        # 获取npu-smi信息
        npu_info = get_npu_smi_info()
        hardware_info["npu_smi_info"] = npu_info
        
        if npu_info:
            print(f"🎯 NPU芯片数量: {npu_info.get('chip_count', 0)}")
            if npu_info.get('chips'):
                for chip in npu_info['chips']:
                    print(f"  芯片 {chip.get('chip_id')}: {chip.get('product_name')}")
                    # print(f"    温度: {chip.get('temp')}°C, 功耗: {chip.get('power')}W")
                    # print(f"    AI Core使用率: {chip.get('ai_core_usage')}%")
    except ImportError:
        print("⚠️  未检测到昇腾npu环境")
    
    
    return hardware_info

def get_npu_smi_info() -> Optional[Dict[str, Any]]:
    """
    获取npu-smi info信息
    返回解析后的NPU状态信息
    """
    try:
        # 尝试执行npu-smi info命令
        result = subprocess.run(
            ['npu-smi', 'info'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            # 尝试其他可能的命令格式
            result = subprocess.run(
                ['npu-smi'],
                capture_output=True,
                text=True,
                timeout=5
            )
        
        if result.returncode == 0:
            return parse_npu_smi_output(result.stdout)
        else:
            print(f"❌ npu-smi命令执行失败: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("❌ 未找到npu-smi命令，请确认昇腾驱动已安装")
        return None
    except subprocess.TimeoutExpired:
        print("❌ npu-smi命令执行超时")
        return None
    except Exception as e:
        print(f"❌ 获取npu-smi信息失败: {e}")
        return None

def parse_npu_smi_output(output: str) -> Dict[str, Any]:
    """
    解析npu-smi info的输出
    支持多种输出格式
    """
    npu_info = {
        'chip_count': 0,
        'chips': [],
        'driver_version': '',
        'firmware_version': '',
        'timestamp': ''
    }
    
    try:
        lines = output.strip().split('\n')
        
        # 查找驱动和固件版本
        for line in lines:
            # 驱动版本
            if 'Driver Version' in line:
                match = re.search(r'Driver Version\s*:\s*([\d\.]+)', line)
                if match:
                    npu_info['driver_version'] = match.group(1)
            
            # 固件版本
            elif 'Firmware Version' in line:
                match = re.search(r'Firmware Version\s*:\s*([\d\.]+)', line)
                if match:
                    npu_info['firmware_version'] = match.group(1)
            
            # 时间戳
            elif 'Timestamp' in line:
                match = re.search(r'Timestamp\s*:\s*(.+)', line)
                if match:
                    npu_info['timestamp'] = match.group(1).strip()
        
        # 查找芯片信息表格
        chip_start = -1
        for i, line in enumerate(lines):
            if 'NPU' in line and ('Temp' in line or 'Temperature' in line) and ('Power' in line or '功耗' in line):
                chip_start = i
                break
        
        chips = []
        
        if chip_start != -1 and chip_start + 1 < len(lines):
            # 表头行
            header_line = lines[chip_start]
            # 数据行（可能是多行）
            for i in range(chip_start + 1, len(lines)):
                data_line = lines[i].strip()
                if not data_line or '===' in data_line or '---' in data_line:
                    continue
                
                # 尝试多种分隔符
                if '|' in data_line:
                    parts = [p.strip() for p in data_line.split('|') if p.strip()]
                else:
                    parts = data_line.split()
                
                if len(parts) >= 6:
                    chip = {
                        'chip_id': parts[0] if len(parts) > 0 else '0',
                        'product_name': parts[1] if len(parts) > 1 else 'Unknown',
                        'health': parts[2] if len(parts) > 2 else 'Unknown',
                        'temp': extract_number(parts[3]) if len(parts) > 3 else '0',
                        'power': extract_number(parts[4]) if len(parts) > 4 else '0',
                        'ai_core_usage': extract_number(parts[5]) if len(parts) > 5 else 0.0,
                        'memory_usage': extract_number(parts[6]) if len(parts) > 6 else 0.0
                    }
                    chips.append(chip)
        
        # 如果没有找到表格格式，尝试其他格式
        if not chips:
            # 尝试匹配常见的npu-smi输出模式
            chip_pattern = re.compile(r'NPU\s*(\d+)\s*:\s*(.+?)\s*Temperature\s*:\s*(\d+)°C\s*Power\s*:\s*(\d+)W\s*AI Core\s*:\s*(\d+)%\s*Memory\s*:\s*(\d+)%')
            for line in lines:
                match = chip_pattern.search(line)
                if match:
                    chip = {
                        'chip_id': match.group(1),
                        'product_name': match.group(2).strip(),
                        'health': 'Good',
                        'temp': match.group(3),
                        'power': match.group(4),
                        'ai_core_usage': float(match.group(5)),
                        'memory_usage': float(match.group(6))
                    }
                    chips.append(chip)
        
        # 如果仍然没有找到，尝试简单统计
        if not chips:
            # 检查/dev/davinci设备数量
            if os.path.exists('/dev'):
                import glob
                davinci_devices = glob.glob('/dev/davinci*')
                control_devices = [d for d in davinci_devices if re.search(r'davinci\d+$', d)]
                
                for i, device in enumerate(control_devices):
                    chip_id = re.search(r'davinci(\d+)', device)
                    if chip_id:
                        chip = {
                            'chip_id': chip_id.group(1),
                            'product_name': 'Ascend NPU',
                            'health': 'Unknown',
                            'temp': 'N/A',
                            'power': 'N/A',
                            'ai_core_usage': 0.0,
                            'memory_usage': 0.0
                        }
                        chips.append(chip)
        
        npu_info['chip_count'] = len(chips)
        npu_info['chips'] = chips
        
        return npu_info
        
    except Exception as e:
        print(f"❌ 解析npu-smi输出失败: {e}")
        return npu_info

def extract_number(text: str) -> float:
    """从文本中提取数字"""
    if not text:
        return 0.0
    
    # 移除非数字字符（保留小数点和负号）
    cleaned = re.sub(r'[^\d\.\-]', '', text)
    if cleaned:
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0
    
   