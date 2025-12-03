#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化配置向导
帮助用户根据系统资源选择合适的配置方案
"""

import os
import sys
import psutil
from pathlib import Path


def get_system_info():
    """获取系统资源信息"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_available else 0
    except:
        cuda_available = False
        gpu_memory = 0
    
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    return {
        'ram_gb': ram_gb,
        'cuda_available': cuda_available,
        'gpu_memory_gb': gpu_memory
    }


def recommend_config(system_info):
    """根据系统资源推荐配置"""
    ram = system_info['ram_gb']
    has_gpu = system_info['cuda_available']
    gpu_mem = system_info['gpu_memory_gb']
    
    print(f"\n{'='*60}")
    print("系统资源检测结果：")
    print(f"{'='*60}")
    print(f"💾 系统内存: {ram:.1f} GB")
    print(f"🎮 GPU 状态: {'可用' if has_gpu else '不可用'}")
    if has_gpu:
        print(f"📊 GPU 显存: {gpu_mem:.1f} GB")
    print(f"{'='*60}\n")
    
    # 推荐配置
    if ram < 8 or (not has_gpu):
        config_type = "低内存方案"
        config = {
            'STUDENT_LOAD_IN_4BIT': '1',
            'STUDENT_FORCE_CPU': '1',
            'ENABLE_HYBRID': '1',
        }
        notes = "使用 4bit 量化 + CPU 模式，内存占用约 2-3GB，速度较慢但稳定"
    elif ram < 16:
        config_type = "标准方案"
        config = {
            'STUDENT_LOAD_IN_4BIT': '1',
            'STUDENT_FORCE_CPU': '0',
            'ENABLE_HYBRID': '1',
        }
        notes = "使用 4bit 量化 + GPU 模式，内存占用约 3-5GB，性能良好"
    else:
        config_type = "高性能方案"
        config = {
            'STUDENT_LOAD_IN_4BIT': '0',
            'STUDENT_FORCE_CPU': '0',
            'ENABLE_HYBRID': '1',
        }
        notes = "使用 FP16 + GPU 模式，内存占用约 6-10GB，性能最佳"
    
    return config_type, config, notes


def update_env_file(config):
    """更新 .env 文件"""
    env_path = Path(__file__).parent / '.env'
    env_example = Path(__file__).parent / '.env.example'
    
    # 如果 .env 不存在，从 .env.example 复制
    if not env_path.exists() and env_example.exists():
        print(f"📝 从 .env.example 创建 .env 文件...")
        with open(env_example, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # 读取现有配置
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    else:
        lines = []
    
    # 更新配置
    updated_lines = []
    updated_keys = set()
    
    for line in lines:
        updated = False
        for key, value in config.items():
            if line.strip().startswith(key + '='):
                updated_lines.append(f"{key}={value}\n")
                updated_keys.add(key)
                updated = True
                break
        if not updated:
            updated_lines.append(line)
    
    # 添加缺失的配置
    for key, value in config.items():
        if key not in updated_keys:
            updated_lines.append(f"{key}={value}\n")
    
    # 写回文件
    with open(env_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"✅ 配置已更新到 {env_path}")


def main():
    print("\n" + "="*60)
    print("🔧 多智能体学术写作系统 - 内存优化配置向导")
    print("="*60)
    
    # 检测系统资源
    print("\n⏳ 正在检测系统资源...")
    system_info = get_system_info()
    
    # 推荐配置
    config_type, config, notes = recommend_config(system_info)
    
    print(f"📋 推荐配置方案: {config_type}")
    print(f"📝 说明: {notes}\n")
    print("推荐配置项：")
    for key, value in config.items():
        print(f"  {key}={value}")
    
    # 询问是否应用
    print(f"\n{'='*60}")
    response = input("是否应用此配置到 .env 文件? (y/n, 默认 y): ").strip().lower()
    
    if response in ('', 'y', 'yes'):
        update_env_file(config)
        print("\n✅ 配置完成！")
        print("\n下一步：")
        print("1. 检查 .env 文件中的 API 密钥等其他配置")
        print("2. 运行 'python web_interface/start_web.py' 启动应用")
    else:
        print("\n❌ 已取消配置")
        print("您可以手动编辑 .env 文件进行配置")
    
    print(f"\n{'='*60}")
    print("💡 提示：")
    print("  - 如果仍然遇到内存问题，可以设置 ENABLE_HYBRID=0")
    print("  - 查看 MEMORY_OPTIMIZATION.md 了解更多优化方案")
    print("  - 使用 FORCE_STUDENT_STUB=1 可以快速测试（不加载真实模型）")
    print("="*60 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 用户取消操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
