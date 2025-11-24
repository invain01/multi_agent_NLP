# 多智能体学术写作优化系统 - 项目结构总览

## 📁 完整项目结构

```
multi_agent_NLP/                          # 项目根目录
├── 📄 README.md                          # 项目主要说明文档
├── 📄 requirements.txt                   # Python依赖包列表
├── 📄 multi_agent_nlp_project.py         # 核心多智能体系统实现
├── 📄 multi_agent_nlp_project.ipynb      # Jupyter Notebook版本
├── 📄 lora_distill.py                    # LoRA微调脚本
├── 📄 paper_draft.txt                    # 论文草稿示例
├── 📄 start_web.bat                      # Windows快速启动脚本
├── 📄 start_web.sh                       # Linux/macOS快速启动脚本
├── 📄 PROJECT_STRUCTURE.md               # 本文件
├── 🗂️  web_interface/                    # 🌟 Web图形化界面
│   ├── 📄 app.py                         # Flask后端应用
│   ├── 📄 start_web.py                   # Web服务启动脚本
│   ├── 📄 demo.py                        # 演示脚本
│   ├── 📄 index.html                     # 主页面
│   ├── 📄 requirements_web.txt           # Web界面依赖
│   ├── 📄 README.md                      # Web界面详细说明
│   └── 🗂️  static/                       # 静态资源
│       ├── 🗂️  css/
│       │   └── 📄 styles.css             # 样式文件
│       └── 🗂️  js/
│           └── 📄 app.js                 # 前端应用逻辑
├── 🗂️  data/                             # 数据目录(运行时创建)
├── 🗂️  output/                           # 输出目录
├── 🗂️  tests/                            # 测试文件
│   └── 📄 test_flow.py
├── 🗂️  GPT_API_free/                     # 免费API相关
│   ├── 📄 demo.py
│   ├── 📄 LICENSE
│   ├── 📄 README.md
│   └── 🗂️  images/
└── 🗂️  __pycache__/                      # Python缓存目录
```

## 🚀 使用方式

### 方式1: Web图形界面 (推荐)
```bash
# Windows
start_web.bat

# Linux/macOS  
./start_web.sh

# 手动启动
cd web_interface
python start_web.py
```

### 方式2: 命令行界面
```bash
# 文本优化
python multi_agent_nlp_project.py demo --text "您的文本"

# 长文件优化  
python multi_agent_nlp_project.py demo --text-file "文件路径"

# 数据合成
python multi_agent_nlp_project.py synthesize --seeds-file "种子文件"

# 评估分析
python multi_agent_nlp_project.py eval
```

### 方式3: Jupyter Notebook
```bash
jupyter notebook multi_agent_nlp_project.ipynb
```

## 🌟 主要功能

### 核心功能
- ✅ **多智能体协作**: Agent A优化 + Agent B评审的双智能体系统
- ✅ **实时进度显示**: WebSocket实时反馈优化过程
- ✅ **长文本处理**: 支持自动分段处理长文档  
- ✅ **多格式导出**: 文本、HTML报告、JSON数据
- ✅ **配置管理**: 可视化API配置界面

### 高级功能
- ✅ **数据合成**: 批量生成训练数据
- ✅ **评估分析**: 多维度性能指标评估
- ✅ **数据蒸馏**: 生成监督学习训练对
- ✅ **LoRA微调**: 支持小模型微调
- ✅ **工具调用**: 集成搜索、代码执行等工具

## 🎯 适用场景

### 学术写作优化
- 论文草稿优化
- 学术报告改进
- 研究提案润色
- 文献综述完善

### 教学与研究
- 多智能体系统研究
- 自然语言处理教学
- 文本生成模型训练
- 学术写作辅助工具开发

### 技术实验
- Prompt工程实验
- 模型微调实验
- 评估指标研究
- 人机协作研究

## 📊 技术栈

### 后端
- **核心**: Python 3.8+, LangChain, OpenAI API
- **Web**: Flask, Flask-SocketIO, Flask-CORS
- **数据**: FAISS, NumPy, Pandas
- **微调**: Transformers, PEFT, LoRA

### 前端  
- **框架**: Bootstrap 5, Vanilla JavaScript
- **通信**: Socket.IO, RESTful API
- **样式**: 自定义CSS, 响应式设计

### 存储
- **向量存储**: FAISS + OpenAI Embeddings
- **数据格式**: JSON, JSONL, HTML
- **配置**: 环境变量 + 浏览器本地存储

## 🔧 配置要求

### 必需配置
```env
OPENAI_API_KEY=your_openai_api_key         # 必需
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选
LLM_MODEL=gpt-4o-mini                      # 可选
```

### 可选配置  
```env
SERPAPI_API_KEY=your_serpapi_key           # 网络搜索功能
EMBED_MODEL_NAME=text-embedding-3-small   # 嵌入模型
```

## 📖 文档说明

- **📄 README.md**: 主要项目说明和使用指南
- **📄 web_interface/README.md**: Web界面详细使用说明  
- **📄 multi_agent_nlp_project.py**: 核心代码，包含详细注释
- **📄 multi_agent_nlp_project.ipynb**: 交互式使用示例

## 🤝 贡献指南

1. Fork项目仓库
2. 创建功能分支: `git checkout -b feature/新功能`
3. 提交更改: `git commit -am '添加新功能'`
4. 推送分支: `git push origin feature/新功能`
5. 提交Pull Request

## 📄 许可证

本项目采用开源许可证，具体详情请查看LICENSE文件。

---
*最后更新: 2024年11月*