# 多智能体 NLP 项目：学术写作优化系统

本项目构建了一个集成了大脑、规划、工具和记忆等组件的复杂双智能体学术写作优化系统，包含知识蒸馏与数据合成功能。

## 📋 项目简介

这是一个基于 LangChain 框架的多智能体 NLP 系统，专门用于学术论文的写作优化。系统采用双智能体协作架构，通过 Agent A（学术表达优化者）和 Agent B（学术批评者）的迭代协作，实现高质量的学术文本优化。

### 核心特性

- 🤖 **双智能体协作系统**: Agent A 负责优化，Agent B 负责批评与改进建议
- 🧠 **知识蒸馏系统**: 通过专家模型指导，快速提升智能体的学术写作能力
- 💾 **知识持久化**: 使用 FAISS 向量数据库存储和检索历史知识
- 🔧 **丰富工具集**: 网络搜索、代码执行、文件读写等工具支持
- 🎯 **交互式界面**: 友好的用户交互界面，支持多种优化模式

## 🏗️ 系统架构

### 核心组件

1. **环境设置**: 安装必要的 Python 库
2. **API 密钥配置**: 安全地加载 API 密钥
3. **智能体架构**:
   - **大脑**: 集成 GPT-4o-mini LLM
   - **工具**: 定义网络搜索、代码执行和文件操作等工具
   - **记忆**: 使用 FAISS 向量数据库实现长期记忆
   - **双智能体系统**: Agent A (优化者) 和 Agent B (批评者) 协作
4. **知识蒸馏**: 专家指导与 Agent 学习系统
5. **持久化知识**: 知识存储与动态应用机制

### 双智能体系统

#### 🤖 Agent A (学术表达优化者)
- **职责**: 接收用户文本，进行学术化改写和表达优化
- **能力**: 学术化表达、逻辑清晰度提升、语言正式性增强
- **输出**: 优化版本 + 修改策略说明

#### 🎓 Agent B (学术批评与改进建议者)
- **职责**: 严格审视 Agent A 的修改，提出改进建议
- **能力**: 概念清晰度评估、学术规范性检查、逻辑严谨性审查
- **输出**: 具体改进建议 + 优先级排序

#### 🔄 协作流程
1. 用户提供原文 + 修改要求
2. 多轮迭代：A优化→B审视→A再优化→B再审视...
3. 生成最终版本 + 完整训练数据

## 🚀 快速开始

### 环境要求

- Python 3.8+
- OpenAI API 密钥
- SerpAPI 密钥（用于网络搜索功能）

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd multi_agent_NLP
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置 API 密钥**

在项目根目录创建 `.env` 文件，添加以下内容：
```
OPENAI_API_KEY="your_openai_api_key"
SERPAPI_API_KEY="your_serpapi_api_key"
```

**注意**: 确保 `.env` 文件已添加到 `.gitignore` 中，不要将 API 密钥提交到版本控制系统。

4. **运行项目**

打开 Jupyter Notebook：
```bash
jupyter notebook multi_agent_nlp_project.ipynb
```

按照 notebook 中的顺序执行各个代码单元格。

### 使用示例

```python
# 启动交互式会话
interactive_viewer.run_interactive_session()

# 查看知识库统计信息
persistent_knowledge.display_knowledge_stats()

# 查看会话历史
interactive_viewer.show_session_history()
```

## 📦 依赖项

主要依赖包包括：

- `openai`: OpenAI API 客户端
- `langchain`: LangChain 框架核心
- `langchain-openai`: LangChain OpenAI 集成
- `langchain-community`: LangChain 社区工具
- `langchain-experimental`: LangChain 实验性功能
- `langchain-core`: LangChain 核心组件
- `google-search-results`: SerpAPI 搜索工具
- `faiss-cpu`: FAISS 向量数据库（CPU 版本）
- `tiktoken`: OpenAI 文本分词工具
- `python-dotenv`: 环境变量管理

详细依赖列表请参见 `requirements.txt`。

## 🎯 功能说明

### 1. 双智能体协作优化

系统通过多轮迭代优化学术文本：
- **快速模式**: 3 轮迭代
- **标准模式**: 5 轮迭代（默认）
- **深度模式**: 8 轮迭代

### 2. 知识蒸馏

通过专家级模型指导，让 Agent A 快速学习学术写作专业知识：
1. Agent A 先尝试优化
2. 专家模型评估并提供指导
3. Agent A 根据指导学习改进
4. Agent B 验证学习效果

### 3. 知识持久化

- 使用 FAISS 向量数据库存储历史知识
- 支持相似性检索，自动应用相关历史经验
- 动态增强智能体的知识库

### 4. 交互式界面

提供友好的用户交互界面，支持：
- 多种优化要求选择（语法检查、逻辑优化、学术表达提升等）
- 多语言支持（中文/英文）
- 实时查看迭代过程
- 导出训练数据

## 📝 使用说明

### 基本使用流程

1. **启动系统**: 按照 notebook 顺序执行所有初始化代码单元格
2. **输入文本**: 通过交互式界面输入需要优化的学术文本
3. **选择要求**: 选择优化要求（可多选）
4. **选择模式**: 选择优化轮数（快速/标准/深度）
5. **查看结果**: 系统会显示每轮迭代的详细过程和最终结果

### 优化要求选项

- 1️⃣ 语法检查与修正
- 2️⃣ 逻辑结构优化
- 3️⃣ 学术表达提升
- 4️⃣ 参考文献格式化
- 5️⃣ 创新性增强
- 6️⃣ 可读性改进
- 7️⃣ 自定义要求

## 🔧 配置说明

### API 配置

项目使用 ChatAnywhere 提供的 OpenAI API 转发服务（国内访问更稳定）：
- Base URL: `https://api.chatanywhere.tech/v1`
- 模型: `gpt-4o-mini`（200次/天免费额度）

### 向量数据库配置

- 嵌入模型: OpenAI `text-embedding-ada-002`
- 向量维度: 1536
- 存储方式: FAISS IndexFlatL2

## 📊 项目结构

```
multi_agent_NLP/
├── multi_agent_nlp_project.ipynb  # 主项目文件
├── requirements.txt                # 依赖列表
├── README.md                       # 项目说明文档
├── .env                            # API 密钥配置（需自行创建）
└── .gitignore                      # Git 忽略文件
```

## ⚠️ 注意事项

1. **API 密钥安全**: 请勿将 `.env` 文件提交到版本控制系统
2. **API 调用限制**: 注意 API 调用频率限制，避免超出配额
3. **网络连接**: 需要稳定的网络连接以访问 API 服务
4. **依赖版本**: 建议使用虚拟环境管理依赖，避免版本冲突

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- LangChain 框架
- OpenAI GPT 模型
- FAISS 向量数据库
- ChatAnywhere API 转发服务
