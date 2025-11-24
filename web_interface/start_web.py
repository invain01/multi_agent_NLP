#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多智能体学术写作优化系统 Web界面启动脚本
"""

import sys
import os
import subprocess
from pathlib import Path

# 添加上级目录到Python路径，以便导入主项目模块
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# 加载环境变量 - 尝试多个可能的.env文件位置
from dotenv import load_dotenv

# 尝试加载.env文件（优先级：当前目录 -> 上级目录）
env_paths = [
    Path(__file__).parent / '.env',  # web_interface/.env
    parent_dir / '.env',             # project_root/.env
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ 已加载环境配置: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    print("⚠️  未找到.env文件，将使用系统环境变量或Web配置")

def check_requirements():
    """检查依赖是否安装"""
    try:
        import flask
        import flask_cors
        import flask_socketio
        print("✅ 基础Web依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少Web依赖: {e}")
        print("请运行: pip install -r requirements_web.txt")
        print(f"当前目录: {Path.cwd()}")
        return False

def check_project_structure():
    """检查项目结构"""
    required_files = [
        '../multi_agent_nlp_project.py',
        'app.py',
        'index.html',
        'static/css/styles.css',
        'static/js/app.js'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        return False
    
    print("✅ 项目结构检查通过")
    return True

def setup_directories():
    """创建必要的目录"""
    directories = ['data', 'static', 'static/css', 'static/js', 'uploads']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✅ 目录结构已创建")

def check_environment():
    """检查环境配置"""
    env_vars = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'OPENAI_BASE_URL': os.getenv('OPENAI_BASE_URL'),
        'LLM_MODEL': os.getenv('LLM_MODEL'),
    }
    
    missing_vars = []
    configured_vars = []
    
    for var, value in env_vars.items():
        if not value:
            missing_vars.append(var)
        else:
            configured_vars.append(f"{var}={'*' * min(8, len(value)) if 'KEY' in var else value}")
    
    if configured_vars:
        print("✅ 已配置的环境变量:")
        for var_info in configured_vars:
            print(f"   - {var_info}")
    
    if missing_vars:
        print(f"⚠️  缺少环境变量: {missing_vars}")
        print("可以在Web界面的配置页面中设置，或创建.env文件")
    else:
        print("✅ 环境变量配置完整")

def main():
    """主启动函数"""
    print("🚀 多智能体学术写作优化系统 Web启动器")
    print("=" * 50)
    
    # 检查依赖
    if not check_requirements():
        sys.exit(1)
    
    # 检查项目结构
    if not check_project_structure():
        sys.exit(1)
    
    # 设置目录
    setup_directories()
    
    # 检查环境
    check_environment()
    
    print("\n📝 准备启动Web服务器...")
    print("💡 访问地址: http://localhost:5000")
    print("💡 支持的功能:")
    print("   - 学术文本优化")
    print("   - 长文件分段处理") 
    print("   - 数据合成与评估")
    print("   - 实时进度显示")
    print("   - 结果下载与报告生成")
    print("\n按 Ctrl+C 停止服务")
    print("=" * 50)
    
    try:
        # 启动Flask应用
        print("📡 正在初始化Flask应用...")
        from app import app, socketio
        print("✅ Flask应用初始化成功")
        print("🌐 启动Web服务器...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("请确保已安装所有依赖包并且主项目文件存在")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()