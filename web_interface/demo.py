#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Web界面演示脚本
用于展示多智能体学术写作优化系统的Web界面功能
"""

import webbrowser
import time
import os
from pathlib import Path

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    多智能体学术写作优化系统 Web界面                            ║
║                             演示与使用指南                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def print_features():
    print("""
🌟 主要功能特性:

📝 文本优化模块
   ├─ 直接文本输入优化
   ├─ 文件上传处理（支持长文本分段）
   ├─ 实时进度显示
   ├─ 多视图结果展示（对比/详情/轮次）
   └─ 多格式下载（文本/HTML报告/JSON）

🔬 数据合成模块
   ├─ 基于种子文本批量生成数据
   ├─ 可配置优化需求和轮次
   └─ 自动生成JSONL数据集

📊 评估分析模块
   ├─ 多维度性能指标评估
   ├─ 可视化评估结果展示
   └─ 支持自定义测试用例

⚗️  数据蒸馏模块
   ├─ JSONL数据蒸馏处理
   ├─ 生成监督学习训练对
   └─ 支持模型微调数据准备

⚙️  配置管理
   ├─ 可视化API配置界面
   ├─ 环境变量管理
   └─ 本地配置保存
    """)

def print_usage_guide():
    print("""
🚀 使用指南:

1️⃣  环境准备
   • 确保已安装Python 3.8+
   • 安装项目依赖: pip install -r requirements.txt
   • 安装Web依赖: pip install -r requirements_web.txt

2️⃣  配置设置
   • 创建.env文件或使用Web配置界面
   • 设置OPENAI_API_KEY（必需）
   • 设置OPENAI_BASE_URL（可选）
   • 设置LLM_MODEL（可选）

3️⃣  启动服务
   • 运行: python start_web.py
   • 或者: python app.py
   • 访问: http://localhost:5000

4️⃣  基本使用流程
   • 在文本优化页面输入要优化的文本
   • 设置优化需求（如：学术表达提升,逻辑结构优化）
   • 点击"开始优化"按钮
   • 实时查看优化进度
   • 下载优化结果和详细报告
    """)

def create_demo_files():
    """创建演示用的示例文件"""
    demo_dir = Path("demo_data")
    demo_dir.mkdir(exist_ok=True)
    
    # 创建示例文本文件
    sample_text = """
这是一个关于机器学习在自然语言处理中应用的研究论文初稿。本文试图探讨深度学习模型在文本分类任务中的性能表现。

我们的研究表明，使用预训练的语言模型可以显著提升文本分类的准确率。实验结果显示，BERT模型在我们的数据集上取得了90%的准确率，这比传统的机器学习方法高出了15个百分点。

然而，我们也发现了一些问题。首先，模型的训练时间较长，需要大量的计算资源。其次，模型的可解释性较差，难以理解模型的决策过程。

未来的研究方向包括：提高模型的效率、增强模型的可解释性、以及探索更加轻量化的模型架构。
    """.strip()
    
    sample_file = demo_dir / "sample_academic_text.txt"
    sample_file.write_text(sample_text, encoding='utf-8')
    
    # 创建种子文本文件
    seed_texts = """
本研究探讨了基于多智能体的文本优化框架，初步实验尚不充分。
我们提出一个简单的管线，但方法部分缺乏清晰的因果论证。
实验结果显示一定改进，但统计显著性需要进一步说明。
文献综述部分相对薄弱，缺少对相关工作的深入分析。
数据集的规模较小，泛化能力有待验证。
    """.strip()
    
    seeds_file = demo_dir / "seed_texts.txt"
    seeds_file.write_text(seed_texts, encoding='utf-8')
    
    print(f"📄 演示文件已创建在: {demo_dir}")
    print(f"   - {sample_file}")
    print(f"   - {seeds_file}")

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    import sys
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    # 检查依赖
    required_packages = [
        'flask', 'flask_cors', 'flask_socketio', 
        'langchain', 'requests', 'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {missing_packages}")
        print("请运行: pip install -r requirements_web.txt")
        return False
    
    print("✅ 环境检查通过")
    return True

def print_demo_scenarios():
    print("""
🎯 演示场景建议:

场景1: 基础文本优化
  • 使用demo_data/sample_academic_text.txt
  • 设置需求: "学术表达提升,逻辑结构优化"
  • 观察实时优化过程和结果对比

场景2: 长文本分段处理
  • 上传更长的文档文件
  • 调整分段设置（如分段大小5000，重叠200）
  • 体验分段优化功能

场景3: 数据合成演示
  • 使用demo_data/seed_texts.txt中的种子文本
  • 设置合成需求和轮次
  • 生成训练数据集

场景4: 评估分析
  • 输入多个测试用例
  • 运行性能评估
  • 查看详细指标报告

场景5: 配置管理
  • 在配置界面设置API参数
  • 测试不同模型的效果
  • 体验配置保存功能
    """)

def main():
    print_banner()
    print_features()
    
    # 检查环境
    if not check_environment():
        return
    
    # 创建演示文件
    create_demo_files()
    
    print_usage_guide()
    print_demo_scenarios()
    
    print("""
🎉 准备就绪！现在可以：

1. 运行启动脚本: python start_web.py
2. 或直接启动应用: python app.py
3. 访问Web界面: http://localhost:5000

💡 提示: 
   • 首次使用请在配置页面设置API密钥
   • 可以使用演示文件快速体验功能
   • 查看浏览器控制台获取详细日志信息
   • 支持多标签页同时使用

📚 更多信息请参考: README.md

按 Enter 键打开Web界面...
    """)
    
    input()
    
    # 尝试打开浏览器
    try:
        webbrowser.open('http://localhost:5000')
        print("🌐 正在打开Web界面...")
    except Exception as e:
        print(f"⚠️  无法自动打开浏览器: {e}")
        print("请手动访问: http://localhost:5000")

if __name__ == '__main__':
    main()