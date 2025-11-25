# 多智能体学术写作优化系统（Academic Multi-Agent WritingOptimizer）

本项目实现了一个面向学术写作的多智能体闭环系统：

- Agent A（学生 / Optimizer）：在原始草稿基础上进行学术化改写和结构优化，可使用**本地 Qwen 学生模型 + LoRA**。
- Agent B（教师 / Reviewer）：基于高质量 LLM（如 DeepSeek）给出评分和改进建议，也可在无 API 的情况下退化为 DummyLLM。
- 多轮交互后生成：优化后的文本 + 每轮改写与评分日志 + HTML / JSON 报告，可用于**论文打磨、教学展示、数据合成与蒸馏、小模型增强**等。

---
## 目录
1. 项目概览与目标  
2. 核心特性速览  
3. 代码结构与模块职责  
4. 整体流程概览  
5. 环境与依赖（Python 3.11 + GPU + 本地模型）  
6. 从零安装与启动  
7. 本地 Qwen + LoRA 学生模型  
8. CLI 模式：demo / synthesize / distill / eval  
9. 多智能体协作机制（Agent A / Agent B）  
10. 工具调用与回退策略（Search / Python REPL / 文件 IO）  
11. 向量记忆与检索（FAISS / 简化回退）  
12. 学术质量评估指标体系  
13. 数据管线：seeds → synth_*.jsonl → distill_pairs.jsonl → LoRA  
14. Web 界面  
15. 团队协作与 runs/ 使用说明  
16. 测试与质量保障（可选）  

---
## 1. 项目概览与目标

本项目提供一个可本地化部署的**学术写作多智能体系统**：

- Agent A：负责在给定需求下对学术文本进行重写、结构调整、语言学术化处理；
- Agent B：负责从多个维度评价 Agent A 的输出，并提出进一步改进建议；
- 支持长文本自动分段、多轮迭代优化、可视化报告生成；
- 支持基于该流程自动合成数据、构建蒸馏样本，并对本地 Qwen 学生模型进行 LoRA/QLoRA 微调；
- 在缺少外部服务时，自动退化为 DummyLLM、简化向量检索等，占位但不断流，适合教学演示与离线环境。

---
## 2. 核心特性速览

- 多智能体闭环：学生模型负责改写，教师模型负责打分与反馈，多轮迭代收敛到高质量版本；
- 学术质量评估：内置 9 维学术质量指标（规范性、证据完整性、流畅度等），可比较优化前后；
- 本地学生模型：通过 `hf_student_llm.py` 加载本地 Qwen（推荐 Qwen1.5/Qwen2 1.8B Chat），可叠加 LoRA 适配器；
- 数据闭环：支持从 `seeds.txt` 出发自动合成数据（synthesize）、从合成日志提炼蒸馏样本（distill）、再通过 `lora_distill.py` 进行 LoRA 微调；
- 向量记忆：优先使用 FAISS + embeddings，缺失时退化为简单关键词相似度检索；
- 强健回退设计：LangChain 相关依赖、SerpAPI、FAISS、OpenAI API 缺失时自动采用 stub / 占位逻辑，保证主流程可运行；
- Web + CLI：提供命令行入口与 Flask + SocketIO Web 前端，支持实时查看进度与下载报告。

---
## 3. 代码结构与模块职责

根目录关键文件（部分）：

- `multi_agent_nlp_project.py`
  - 项目主入口，提供 CLI 子命令：`demo` / `synthesize` / `distill` / `eval`；
  - 完成：主 LLM（DeepSeek/OpenAI/Dummy）、工具集、向量存储、MemoryManager 初始化；
  - 定义 `DualAgentAcademicSystem`，实现 Agent A/B 协作、长文本切分、日志与 HTML 报告生成等。
- `hf_student_llm.py`
  - HuggingFace + 可选 LoRA 的“学生模型”封装，类名 `HFChatLLM`；
  - 当 `FORCE_STUDENT_STUB=1` 或缺少 torch/transformers 时，自动使用轻量 stub；
  - 在正常模式中，可加载本地 Qwen 基座以及 `runs/` 下的 LoRA 适配器。
- `lora_distill.py`
  - 从蒸馏数据 JSONL 中读取样本，使用 transformers + peft 微调 LoRA/QLoRA；
  - 对 Qwen/Qwen2 风格的模型自动设置合理 `target_modules`，并支持命令行调整 LoRA 超参。
- `metrics.py` / `demo_metrics.py`
  - 学术质量评价指标与演示脚本（单段/对比/可视化）。
- `web_interface/`
  - `app.py` / `start_web.py`：Flask + SocketIO 后端；
  - `index.html` + `static/js/app.js` + `static/css/styles.css`：前端与交互；
  - `uploads/`：上传文件暂存目录。
- `data/`
  - `seeds.txt`：合成任务的种子句子；
  - `synth_*.jsonl`：一次 `synthesize` 运行得到的完整多轮合成/推理日志；
  - `distill_pairs.jsonl`：由 `distill` 生成的用于 LoRA 训练的蒸馏样本。
- `runs/`
  - 例如 `runs/qwen1_8b_lora_v1/`：你训练得到的 LoRA 适配器与训练 checkpoint（见第 15 节）。
- `tests/test_flow.py`
  - 核心逻辑的单元测试：需求解析、文本切分、DummyLLM 流程回退、HTML 报告结构、蒸馏样本生成等。
- `requirements.txt`
  - 统一的依赖清单，涵盖核心 + Web + 微调 + 评估。

---
## 4. 整体流程概览

```text
seeds.txt  ──► synthesize  ──►  synth_*.jsonl  ──►  distill  ──►  distill_pairs.jsonl
                                                 │
                                                 ▼
                                          lora_distill.py
                                                 │
                                                 ▼
    本地 Qwen 基座模型  +  runs/<lora_run>/adapter_*  ──►  学生模型 (HFChatLLM)
                                                 │
                                                 ▼
                 DualAgentAcademicSystem  (Agent A 学生 + Agent B 老师)
                                                 │
                                                 ▼
                                  CLI / Web  多轮优化 + 报告输出
```

---
## 5. 环境与依赖（Python 3.11 + GPU + 本地模型）

推荐环境：

- OS：Windows 10/11 x64；
- Python：3.11.x（用 venv 单独隔离）；
- GPU：NVIDIA RTX 4060 或同级（推荐 ≥ 8GB 显存）；
- CUDA：建议使用与 PyTorch cu121 对应的驱动版本（安装官方最新驱动通常即可）。

`requirements.txt` 已包含：

- LangChain 相关：`langchain*`, `google-search-results`；
- 向量检索：`faiss-cpu`, `numpy`；
- 通用工具：`tiktoken`, `python-dotenv`, `requests`, `safetensors`；
- HF & 微调：`transformers==4.46.0`, `huggingface-hub<1.0`, `peft>=0.10.0`, `datasets`, `accelerate`, `bitsandbytes`, `optimum`；
- Web：`Flask`, `Flask-SocketIO`, `Flask-CORS`, `python-socketio`, `python-engineio`, `Werkzeug`；
- 测试：`pytest`。

> 注意：PyTorch 未在 `requirements.txt` 中固定版本，请按你本地 GPU / CUDA 环境自行安装。

---
## 6. 从零安装与启动

以 Windows + PowerShell 为例：

### 6.1 创建虚拟环境

```powershell
cd D:\Projects\NLP\multi_agent_NLP
python -m venv .venv
.\.venv\Scripts\activate
```

### 6.2 安装 GPU 版 PyTorch（以 CUDA 12.1 为例）

```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"
```

确保 `cuda available: True` 且 `device count >= 1`。

### 6.3 安装项目依赖

```powershell
cd D:\Projects\NLP\multi_agent_NLP
pip install -r requirements.txt
```

若 `bitsandbytes` 或 `faiss-cpu` 安装有警告，一般可以忽略，项目会自动回退到简化实现。

### 6.4 准备 `.env` 文件

若仓库中有 `.env.example`，可复制为：

```powershell
copy .env.example .env
```

然后根据实际情况编辑 `.env`，示例：

```ini
# 可选：若使用 DeepSeek/OpenAI 兼容 API
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-reasoner

# 可选：若使用 SerpAPI 检索
SERPAPI_API_KEY=
EMBED_MODEL_NAME=text-embedding-3-small

ENABLE_INTERACTIVE=0

# 学生模型（本地 Qwen + LoRA），详见第 7 节
STUDENT_BASE_MODEL=D:/Projects/NLP/models/Qwen1.5-1.8B-Chat
STUDENT_LORA_DIR=
STUDENT_MAX_NEW_TOKENS=256
FORCE_STUDENT_STUB=0
```

### 6.5 运行一个最小 demo

```powershell
cd D:\Projects\NLP\multi_agent_NLP
.\.venv\Scripts\activate

python multi_agent_nlp_project.py demo ^
  --rounds 2 ^
  --text "这是一个需要提升学术表达与逻辑清晰度的段落。" ^
  --requirements "学术表达提升;逻辑结构优化" ^
  --html-report demo.html
```

完成后在浏览器中打开 `demo.html` 即可查看完整报告。

### 6.6 启动 Web 界面

```powershell
cd D:\Projects\NLP\multi_agent_NLP
.\.venv\Scripts\activate

cd web_interface
python start_web.py
```

浏览器访问：`http://localhost:5000`。

---
## 7. 本地 Qwen + LoRA 学生模型

### 7.1 准备基座 Qwen 模型

1. 将 Qwen1.5 / Qwen2 1.8B Chat 模型下载到本地，例如：
   `D:/Projects/NLP/models/Qwen1.5-1.8B-Chat`；
2. 确保该路径能被 transformers 的 `from_pretrained` 识别（结构与 HuggingFace 源仓库一致）。

在 `.env` 中配置：

```ini
STUDENT_BASE_MODEL=D:/Projects/NLP/models/Qwen1.5-1.8B-Chat
STUDENT_LORA_DIR=                            # 暂时留空或指向 runs 中的 LoRA
STUDENT_MAX_NEW_TOKENS=256
FORCE_STUDENT_STUB=0
```

`hf_student_llm.HFChatLLM` 会按以下顺序工作：

- 若 `FORCE_STUDENT_STUB=1` 或缺少 torch/transformers，则使用 stub 学生模型（不加载权重，仅占位）；
- 否则从 `STUDENT_BASE_MODEL` 加载基座 Qwen 模型；
- 若 `STUDENT_LORA_DIR` 存在且已安装 peft，则加载 LoRA 适配器叠加在基座之上。

### 7.2 使用已训练好的 LoRA（如 `runs/qwen1_8b_lora_v1`）

你已经从 `data/distill_pairs.jsonl` 微调出 LoRA：

```powershell
python lora_distill.py ^
  --data data/distill_pairs.jsonl ^
  --model D:/Projects/NLP/models/Qwen1.5-1.8B-Chat ^
  --output runs/qwen1_8b_lora_v1 ^
  --batch 2 ^
  --epochs 1 ^
  --lr 5e-5 ^
  --max-length 1024 ^
  --gradient-accum 8 ^
  --fp16
```

训练成功后，在 `.env` 中填入：

```ini
STUDENT_BASE_MODEL=D:/Projects/NLP/models/Qwen1.5-1.8B-Chat
STUDENT_LORA_DIR=D:/Projects/NLP/multi_agent_NLP/runs/qwen1_8b_lora_v1
STUDENT_MAX_NEW_TOKENS=256
FORCE_STUDENT_STUB=0
```

此时，多智能体流程中 Agent A 将使用“基座 Qwen + 你的 LoRA” 作为学生模型。

### 7.3 Stub 学生模型（测试 / CI / 无模型时）

若设置：

```ini
FORCE_STUDENT_STUB=1
```

则会启用轻量 stub：

- 不加载任何本地权重；
- `.invoke()` 返回带有 `[STUB_GENERATION]` 前缀的占位文本；
- 适合在单元测试、无 GPU、无本地模型时验证流程与报告生成逻辑。

---
## 8. CLI 模式：demo / synthesize / distill / eval

所有 CLI 子命令都通过 `multi_agent_nlp_project.py` 提供。

### 8.1 `demo`：单文本 / 长文本多轮优化

```powershell
python multi_agent_nlp_project.py demo ^
  --rounds 2 ^
  --text "这是一个需要提升学术表达与逻辑清晰度的段落。" ^
  --requirements "学术表达提升;逻辑结构优化" ^
  --html-report demo.html
```

支持参数（部分）：

- `--text` / `--text-file`：直接传文本或指定包含文本的文件；
- `--rounds`：多轮协作的轮数；
- `--chunk-size` / `--chunk-overlap` / `--max-chunks`：长文本切分参数；
- `--no-tools` / `--no-memory`：禁用外部工具或记忆；
- `--html-report`：输出 HTML 报告路径。

### 8.2 `synthesize`：从 `seeds.txt` 合成教师样本（synth_*.jsonl）

```powershell
python multi_agent_nlp_project.py synthesize ^
  --rounds 3 ^
  --seeds-file data\seeds.txt ^
  --out data\synth_academic_YYYYMMDD_HHMMSS.jsonl
```

- `data/seeds.txt`：每行一个种子任务，如研究主题、原始段落、写作指令等；
- 输出的 `synth_*.jsonl`：包含多轮 Agent A/B 对话、评分、teacher signal 等完整日志。

### 8.3 `distill`：从合成日志构建蒸馏训练集（distill_pairs.jsonl）

```powershell
python multi_agent_nlp_project.py distill ^
  --distill-src data\synth_academic_YYYYMMDD_HHMMSS.jsonl ^
  --distill-out data\distill_pairs.jsonl
```

- 从复杂的合成日志中提取结构化的样本：`instruction` + `input` + `output` + `scores` 等；
- 这些样本将作为 `lora_distill.py` 的训练数据。

### 8.4 `eval`：批量评估与报告

```powershell
python multi_agent_nlp_project.py eval ^
  --rounds 2 ^
  --report data\eval_report.json ^
  --html-report eval_report.html
```

可对多组样本运行多轮协作与指标评估，输出 JSON 和 HTML 报告，便于比较不同模型或配置的效果。

---
## 9. 多智能体协作机制（Agent A / Agent B）

简要说明：

- Agent A：
  - 输入：原始文本 + 用户需求 + 上一轮评分/反馈 + 记忆检索片段；
  - 输出：新的优化版本；
  - 在 Hybrid 模式下由本地 Qwen + LoRA 驱动。
- Agent B：
  - 输入：原始文本 + 当前候选文本 + 用户需求；
  - 输出：结构化反馈与多维评分；
  - 通常使用远程高质量模型（如 DeepSeek），若无则退化为 DummyLLM 占位。

每一轮都会记录：

- `optimized_text`、`agent_b_feedback`、`scores`；
- `diff`（前后版本差异）和 `tool_observations`；
- `timestamp` 等元信息。HTML 报告会将这些内容以时间轴方式展示。

---
## 10. 工具调用与回退策略

- 工具集合（在 `multi_agent_nlp_project.py` 中初始化）：
  - `SerpAPIWrapper`（若 `SERPAPI_API_KEY` 存在）：实时搜索；
  - `PythonREPL`：执行小段 Python 代码；
  - 文件读写工具：基于内置 `open` 实现；
- 回退逻辑：
  - 未配置 SerpAPI：搜索工具返回占位提示而不中断流程；
  - 缺少 LangChain 相关依赖：使用 stub 实现（同名类但简化逻辑）；
  - 文件读写始终使用 Python 标准库实现。

---
## 11. 向量记忆与检索

- 默认：
  - Embeddings：`OpenAIEmbeddings` 或 DummyEmbeddings；
  - 向量库：`FAISS`（如 `faiss-cpu` 可用时）;
- 回退：
  - 若 FAISS 或相关依赖不可用，则使用内置 `SimpleVectorStore`，基于中英文分词 + Jaccard 相似度进行排序；
- `MemoryManager` 提供：
  - `add_memory(text, metadata)`：写入带时间戳和命名空间的记忆；
  - `recall(query, k)`：按相似度返回最近的若干记忆文本片段，用于增强下一轮 Prompt。

---
## 12. 学术质量评估指标体系

见 `metrics.py` 与 `METRICS_REFERENCE_CARD.txt`，包含但不限于：

- 学术规范性、引文/证据完整性、论证强度、逻辑结构、表达清晰度、语言流畅度、结构完整性等；
- 支持：单文本打分、优化前后对比、加权综合评估、提升率计算。

`demo_metrics.py` 提供示例用法。

---
## 13. 数据管线：`seeds.txt` → `synth_*.jsonl` → `distill_pairs.jsonl` → LoRA

简要职责：

- `data/seeds.txt`：
  - 每行一个“种子任务”或原始段落，是合成数据的起点；
- `data/synth_*.jsonl`：
  - 执行 `synthesize` 子命令后的输出，每行记录一次完整的多轮合成/推理过程；
- `data/distill_pairs.jsonl`：
  - 执行 `distill` 子命令后的输出，每行是一个精简样本（instruction + input + output + scores 等），直接用于 LoRA 微调。

LoRA 训练由 `lora_distill.py` 完成，输出目录例如 `runs/qwen1_8b_lora_v1/`。

---
## 14. Web 界面

- 启动方式见第 6.6 节；
- 提供：
  - 文本输入与需求配置；
  - 实时查看多轮进度与中间结果；
  - 下载最终 HTML 报告；
- 基于 Flask + SocketIO 实现，适合教学演示和交互式使用。

---
## 15. 团队协作与 `runs/` 使用说明

> 这一节回答常见问题：`runs/` 目录里是什么？团队成员如何复用与继续训练？

- `runs/` 目录保存的是：
  - 你训练得到的 **LoRA 适配器权重**（如 `adapter_model.safetensors`）；
  - 训练 checkpoint（`checkpoint-*` 子目录中的优化器状态等）;
  - tokenizer 拷贝和 `RUN_INFO.txt` 等元数据；
- `runs/` **不是完整的 Qwen 基座模型**，而是基座之上的“小增量参数”。

在推理时，学生 Agent A 实际使用的是：

> **本地基座 Qwen 模型（大） + `runs/<run_name>/` 中的 LoRA 适配器（小）**

因此：

- 将本项目上传到 GitHub 时：
  - 通常只提交 `runs/qwen1_8b_lora_v1/` 这类 LoRA 适配器目录；
  - 不提交体积巨大的基座 Qwen 权重（且原始模型 License 往往不鼓励这样做）。
- 团队其他成员使用你的学生模型时：
  1. 自己在本地或共享存储中准备与您相同版本的 Qwen 基座模型；
  2. 克隆你的仓库，包含 `runs/` 目录；
  3. 在各自的 `.env` 中设置：

     ```ini
     STUDENT_BASE_MODEL=<他们本地的 Qwen 基座路径>
     STUDENT_LORA_DIR=<仓库根路径>/runs/qwen1_8b_lora_v1
     ```

  4. 即可获得与你相同的学生模型行为，**无需重新训练 LoRA**。

若需要“在你的基础上继续训练”：

- 简单做法：
  - 使用相同基座 + 你的 `distill_pairs.jsonl` 或新蒸馏数据，再调用 `lora_distill.py` 训练到新的输出目录（如 `runs/qwen1_8b_lora_v2/`）。
- 进阶做法：
  - 可在 `lora_distill.py` 中扩展 `--resume-from-lora` 等参数，从已有 LoRA 适配器继续训练（对本项目当前功能不是必需，可后续扩展）。

---
## 16. 测试与质量保障（可选）

项目包含一部分单元测试，主要覆盖：

- 需求字符串解析；
- 长文本切分与重叠逻辑；
- 无 OpenAI API Key 时的 DummyLLM 回退流程；
- HTML 报告结构完整性；
- 从合成日志构建蒸馏样本的逻辑；
- 学生 stub / 基座模型混合模式构建。

如需在本地运行：

```powershell
cd D:\Projects\NLP\multi_agent_NLP
.\.venv\Scripts\activate
pytest
```

在 CI 或无 GPU 环境下，建议设置：

```powershell
$env:FORCE_STUDENT_STUB='1'
```

以避免加载真实大模型，使用 stub 学生模型测试整体流程与接口。
