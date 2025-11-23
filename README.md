# 多智能体学术写作优化系统 使用手册

> 主题：学术表达优化助手。系统包含两个智能体：Agent A 负责修改优化草稿，Agent B 负责严格审查并给出结构化评分与建议，二者多轮往复后输出最终结果；同时支持数据合成、蒸馏与 LoRA 微调原型。

---
## 目录
1. 项目简介
2. 核心功能与特性
3. 代码结构说明
4. 环境准备与安装
5. 配置说明（.env / 环境变量）
6. 快速上手：常用命令示例
7. CLI 参数与工作流详情
8. Jupyter Notebook 与脚本的一致性说明
9. 典型使用场景与优化思路
10. 数据格式规范（合成 / 蒸馏 / 报告）
11. 评估指标含义
12. 性能与调优建议
13. 常见问题 FAQ
14. 后续扩展建议

---
## 1. 项目简介
本项目实现了一个针对“学术风格文本优化”的多智能体闭环系统：
- **Agent A（Optimizer）**：根据用户需求与历史记忆对文本进行多轮学术化改写；
- **Agent B（Reviewer）**：对 Agent A 的输出进行结构化评估与打分，指出剩余问题并给出下一轮改进建议；
- 通过多轮 A/B 协作，逐步收敛到质量更高的版本，并记录完整的协作日志、评分、差异（Diff）、工具调用观测等信息。

系统还提供：
- **数据合成（synthesize）**：从种子文本出发，批量生成多轮协作数据（含 teacher signal 与评分）；
- **蒸馏数据构造（distill）**：从合成数据中抽取 `instruction/output` 训练对；
- **LoRA / QLoRA 微调脚本（lora_distill.py）**：将蒸馏对用于小模型微调；
- **评估模式（eval）**：对多条文本运行多轮优化并计算一系列轻量指标与 Agent B 评分均值；
- **HTML / JSON 报告**：可视化展示每一轮的文本、Diff、评分与工具调用信息。

该项目适合用于：
- 学术写作场景原型验证；
- prompt 与多智能体协作策略实验；
- 生成蒸馏数据并微调较小的中文/英文语言模型。

---
## 2. 核心功能与特性
- **多轮双 Agent 协作**：`--rounds` 控制轮数，每轮包含一次 Agent A 改写与一次 Agent B 评审；
- **ReAct 风格工具调用**：在满足特定条件时触发网络搜索（SerpAPI）、Python REPL、文件读写等工具；
- **向量记忆 / 召回**：优先使用 FAISS + Embeddings（若依赖或 Key 缺失则回退到简单字符串相似度检索）；
- **健壮的回退链**：LLM / Embeddings / VectorStore / Search 多层回退，保证即使没有 API Key 也能跑通流程（使用 DummyLLM 占位输出）；
- **长文本分段优化**：支持从文件读取长文，按句子智能分段并带重叠区域进行多轮优化，再将各段拼接为整体；
- **数据合成与蒸馏链路**：从多轮协作日志生成适合指令微调的 `instruction/output` 蒸馏对；
- **评估指标**：自动计算长度变化、词汇多样性、重复度下降、可读性、连贯性、句长方差变化、双元组重复变化等指标，并聚合均值；
- **报告生成**：输出结构化 JSON 以及带 Diff 高亮的 HTML 报告，方便分析与展示。

---
## 3. 代码结构说明
- `multi_agent_nlp_project.py`：
  - 整个系统的主入口脚本；
  - 定义 LLM 初始化与回退逻辑、工具集合、向量记忆、双 Agent 协作系统 `DualAgentAcademicSystem`；
  - 提供命令行接口（demo / synthesize / eval / distill），以及长文本分段优化逻辑；
  - 提供 JSON / HTML 报告生成与写入辅助函数。
- `multi_agent_nlp_project.ipynb`：
  - 与脚本逻辑严格同步的 Notebook 版本；
  - 以章节划分方式呈现环境检查、LLM 初始化、工具层、向量记忆、双 Agent 协作、示例调用等；
  - 适合交互试验与可视化展示；如需参考实现细节，可在 Notebook 中逐格查看对应于脚本的实现片段。
- `lora_distill.py`：
  - 使用 Hugging Face Transformers + PEFT + Datasets 对蒸馏对进行 LoRA / QLoRA 微调；
  - 支持普通 LoRA 与 4bit QLoRA（当安装 `bitsandbytes` 且 `--qlora` 打开时）；
  - 输出微调后的权重与训练配置说明 `RUN_INFO.txt`。
- `requirements.txt`：
  - 项目的 Python 依赖清单，覆盖 LangChain、OpenAI 客户端、检索、微调相关库；
- `.env.example`：
  - 环境变量示例配置文件，可复制为 `.env` 并填入自己的 API Key 与服务地址。

---
## 4. 环境准备与安装
### 4.1 Python 版本
建议使用 Python 3.10 及以上版本。

### 4.2 创建虚拟环境（Windows CMD）
```bat
cd D:\Projects\PythonProject\multi_agent_NLP
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Linux / macOS 示例：
```bash
cd /path/to/multi_agent_NLP
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

---
## 5. 配置说明（.env / 环境变量）
在项目根目录创建 `.env` 文件（可从 `.env.example` 复制）：
```ini
OPENAI_API_KEY=你的OpenAI或兼容接口Key（可选）
SERPAPI_API_KEY=你的SerpAPI Key（可选）
OPENAI_BASE_URL=https://api.deepseek.com        # 或其他 OpenAI 兼容接口
LLM_MODEL=deepseek-reasoner                     # 如 gpt-4o-mini / deepseek-chat 等
EMBED_MODEL_NAME=text-embedding-3-small         # 当 Base URL 支持 embeddings 时生效
ENABLE_INTERACTIVE=0                            # 1 开启交互模式，0 关闭
```
说明：
- 未配置 `OPENAI_API_KEY` 时，系统自动使用 `DummyLLM`，输出为占位文本，仅用于调试流程；
- 当 `OPENAI_BASE_URL` 包含 `deepseek.com` 且 `LLM_MODEL` 非官方推荐名字时，脚本会自动规范为 `deepseek-chat` 或 `deepseek-reasoner`；
- 未配置 `SERPAPI_API_KEY` 时，网络搜索工具会以占位实现替代，但接口仍然可调用。

---
## 6. 快速上手：常用命令示例
以下示例假定你已在虚拟环境中并配置好 `.env`。

### 6.1 单条中文文本多轮优化（Demo）
```bat
python multi_agent_nlp_project.py demo
```

自定义文本与需求：
```bat
python multi_agent_nlp_project.py demo ^
  --rounds 3 ^
  --text "这是一段关于多智能体协作进行学术写作优化的初稿。" ^
  --requirements "学术表达提升;逻辑结构优化" ^
  --report demo.json ^
  --html-report demo.html
```

英文模式示例：
```bat
python multi_agent_nlp_project.py demo ^
  --lang en ^
  --text "This is a rough draft about multi-agent collaboration for academic writing." ^
  --requirements "academic polish,logical coherence" ^
  --report demo_en.json ^
  --html-report demo_en.html
```

### 6.2 长文本文件优化
将待优化的论文草稿保存为 `paper_draft.txt`：
```bat
python multi_agent_nlp_project.py demo ^
  --text-file paper_draft.txt ^
  --rounds 2 ^
  --requirements "学术表达提升,结构清晰" ^
  --chunk-size 5000 ^
  --chunk-overlap 200 ^
  --out-text-file optimized_paper.txt ^
  --report long_demo.json ^
  --html-report long_demo.html
```

### 6.3 数据合成 → 蒸馏 → LoRA 微调
1. 合成多轮协作数据：
```bat
python multi_agent_nlp_project.py synthesize ^
  --rounds 3 ^
  --requirements "学术表达提升,结构清晰,可读性增强" ^
  --out data\synth.jsonl
```

2. 构造蒸馏训练对：
```bat
python multi_agent_nlp_project.py distill ^
  --distill-src data\synth.jsonl ^
  --distill-out data\distill_pairs.jsonl
```

3. 使用 `lora_distill.py` 进行 LoRA 微调：
```bat
python lora_distill.py ^
  --data data\distill_pairs.jsonl ^
  --model qwen/Qwen1.5-0.5B ^
  --output runs\qwen-mini-lora ^
  --epochs 1 ^
  --batch 2 ^
  --fp16
```

如需 QLoRA 4bit（需 GPU 与 `bitsandbytes`）：
```bat
python lora_distill.py ^
  --data data\distill_pairs.jsonl ^
  --model qwen/Qwen1.5-1.8B-Chat ^
  --output runs\qwen-lora-4bit ^
  --epochs 1 ^
  --batch 2 ^
  --qlora ^
  --fp16
```

### 6.4 评估模式
```bat
python multi_agent_nlp_project.py eval ^
  --rounds 2 ^
  --requirements "严谨性,逻辑连贯" ^
  --report eval.json ^
  --html-report eval.html
```

控制台会输出类似：
```text
📈 评估汇总: {"len_gain_avg":0.132,"ttr_gain_avg":0.045,"repetition_delta_avg":0.021,"n":2,...}
```

### 6.5 禁用部分组件做消融实验
```bat
python multi_agent_nlp_project.py demo ^
  --rounds 2 ^
  --no-tools ^
  --no-memory
```
这会关闭工具调用和向量记忆，便于对比多智能体策略本身的贡献。

---
## 7. CLI 参数与工作流详情
脚本入口：`multi_agent_nlp_project.py`，总体调用方式：
```text
python multi_agent_nlp_project.py [command] [--options]
```

### 7.1 command 子命令
- `demo`：单条文本多轮优化（支持长文本分段、报告输出）；
- `synthesize`：基于种子文本批量合成多轮协作数据，写入 JSONL；
- `eval`：对多条测试文本运行协作流程，统计多维指标；
- `distill`：从合成的 JSONL 数据中生成蒸馏训练对 JSONL。

### 7.2 通用参数
- `--rounds <int>`：协作轮次（>=1）；
- `--requirements <str>`：用逗号/分号/中文分号分隔的需求列表；
- `--lang zh|en`：语言（会影响默认初稿与默认需求）；
- `--no-tools`：禁用工具调用层；
- `--no-memory`：禁用向量记忆层；
- `--report <path>`：输出 JSON 报告（demo / eval / synthesize / distill 不同模式下结构略有不同）；
- `--html-report <path>`：输出 HTML 可视化报告（demo / eval / 长文本 demo 支持）。

### 7.3 demo 模式特有参数
- `--text <str>`：直接在命令行提供初始文本；
- `--text-file <path>`：从文本文件读取长初稿并自动分段；
- `--chunk-size <int>`：单段最大字符数（默认 5000，<=0 表示不分段）；
- `--chunk-overlap <int>`：相邻分段的重叠字符数（默认 200，用于保持上下文连贯性）；
- `--max-chunks <int>`：限制最多处理的段数（0 表示不限制，仅用于快速实验）；
- `--out-text-file <path>`：将最终优化后的整篇文本写入该路径。

### 7.4 synthesize / distill 特有参数
- `--seeds-file <path>`（synthesize）：从文件读取种子文本（每行一条）。未提供时使用内置 3 条示例；
- `--out <path>`（synthesize）：指定合成 JSONL 输出路径；
- `--distill-src <path>`（distill）：上游合成 JSONL 源；
- `--distill-out <path>`（distill）：蒸馏对 JSONL 输出路径。

---
## 8. Notebook 与脚本的一致性说明
- `multi_agent_nlp_project.ipynb` 与 `multi_agent_nlp_project.py` 在核心逻辑上保持同步：
  - LLM 初始化与三层回退策略；
  - 工具层构造（SerpAPI / Python REPL / 文件读写）；
  - 向量记忆（FAISS 或 SimpleVectorStore 回退）；
  - 双 Agent 协作系统 `DualAgentAcademicSystem`（prompt 设计、评分解析、Diff 计算）；
- Notebook 以“章节 + 代码单元”的形式展开上述实现，方便逐步执行和可视化调试；
- 当你在脚本中修改核心逻辑时，建议同步更新 Notebook 中对应单元，以保持示例与实际行为一致（当前仓库已完成一次同步校验）。

在 Notebook 中进行一次端到端示例：
1. 运行环境检查与 LLM 初始化单元；
2. 运行工具层与向量记忆单元；
3. 实例化 `DualAgentAcademicSystem`：
   ```python
   system = DualAgentAcademicSystem(llm, TOOLS, vectorstore)
   ```
4. 调用多轮协作：
   ```python
   final_text, log = system.collaborate(
       "这是一段关于多智能体协作进行学术写作优化的初稿。",
       ["学术表达提升", "逻辑结构优化"],
       rounds=2,
   )
   ```
5. 根据 `log` 中的 `scores`、`diff`、`tool_observations` 进一步分析每一轮表现。

---
## 9. 典型使用场景与优化思路
- **学术论文段落打磨**：将 Methods/Results/Discussion 段落依次送入系统，根据“严谨性”“逻辑连贯”“可读性”等需求进行多轮优化；
- **数据生成与蒸馏**：使用较强的商业模型（如 GPT-4 级别）进行多轮协作，生成高质量 teacher signal，再对开源模型、企业私有模型做蒸馏微调；
- **多智能体策略实验**：基于本框架尝试不同的 Agent 角色、评分维度、工具触发逻辑，观察其对最终文本质量与指标的影响；
- **长篇报告 / 毕业论文草稿润色**：利用 `--text-file` + 分段重叠设计，处理较长的中文/英文学术文档，并输出整体优化结果与分段日志。

---
## 10. 数据格式规范
### 10.1 合成数据（synthesize 输出 JSONL）
每行一个 JSON 对象，示例结构：
```json
{
  "id": "case_0",
  "input": "原始种子文本",
  "requirements": ["学术表达提升", "结构清晰"],
  "final": "最终优化文本",
  "log": [
    {"round": 0, "user_input": "...", "requirements": ["..."], "timestamp": "..."},
    {"round": 1, "optimized_text": "...", "agent_b_feedback": "...", "scores": {"quality": 8.0, "rigor": 7.0}, "diff": "...", ...},
    {"round": 2, ...}
  ],
  "created_at": "ISO 时间",
  "teacher_signal": "最后一轮 optimized_text（用于蒸馏）",
  "scores": {"quality": 8.0, "rigor": 7.0, "logic": 7.0, "novelty": 6.0}
}
```

### 10.2 蒸馏对（distill 输出 JSONL）
```json
{"instruction": "优化以下学术段落，满足需求: 学术表达提升, 结构清晰\n原文: ...", "output": "teacher 信号文本", "scores": {"quality":8.0,"rigor":7.0}}
```

### 10.3 LoRA 训练拼接格式
`lora_distill.py` 会将每条 `instruction/output` 拼接为：
```text
指令:
<instruction>

优质答案:
<output>
```
用于常规自回归语言模型微调。

### 10.4 报告 JSON / HTML
- JSON 报告（`--report`）：
  - demo 模式：`{"final": 最终文本, "log": 协作日志}`；
  - 长文本模式：`{"final": 最终整篇文本, "aggregated": {"segments": [...], ...}}`；
  - eval 模式：`{"summary": 指标均值, "cases": 单条样本指标}`；
  - synthesize / distill 模式：包含生成文件路径等元信息。
- HTML 报告（`--html-report`）：
  - 按轮展示优化文本、Agent B 反馈、评分、Diff（新内容高亮为绿色，删除为红色）、工具调用观测；
  - eval 模式下展示指标汇总表格与各案例的概览。

---
## 11. 评估指标含义
| 指标 | 说明 |
|------|------|
| `len_gain` | 长度变化比例 `(final_len - orig_len) / orig_len` |
| `ttr_gain` | 词汇多样性 Type Token Ratio 提升 |
| `repetition_delta` | 最高频 token 占比下降（越大越少重复） |
| `readability_gain` | 基于平均句长的可读性代理值变化 |
| `coherence_gain` | 邻接句子词汇交集/并集比平均提升，作为连贯性代理 |
| `sent_var_delta` | 原句长方差 - 新句长方差（正值表示更均匀） |
| `bigram_rep_delta` | 高频二元组占比下降（重复模式减少） |
| `quality/rigor/logic/novelty` | Agent B 在 JSON 中给出的主观评分均值 |

这些指标由 `DualAgentAcademicSystem.evaluate()` 计算并在 eval 模式中聚合输出。

---
## 12. 性能与调优建议
| 目标 | 建议 |
|------|------|
| 提升速度 | 降低 `--rounds`、缩短初稿、在不关键场景下关闭工具与记忆（`--no-tools --no-memory`） |
| 降低显存占用 | 使用 QLoRA 4bit（`--qlora`），减小 `--max-length`，调低 batch 或使用梯度累积 |
| 强化多样性 | 在合成阶段丰富需求组合与种子语料、多语言混合、增加轮数 |
| 提升 teacher 质量 | 使用更强的上游大模型生成合成数据，再对开源模型、企业私有模型做蒸馏 |
| 减少指令截断 | 在 `lora_distill.py` 中适当提高 `--max-length`（需平衡显存） |
| 训练更平稳 | 增大 `--epochs`、设置合理的 `--warmup-ratio`、按数据规模调整 `--gradient-accum` |

---
## 13. 常见问题 FAQ
1. **导入 faiss 失败？**  
   - 已自动回退到 `SimpleVectorStore`，功能仍可使用；如需更高召回质量与性能，可安装 `faiss-cpu`。

2. **搜索结果总是占位？**  
   - 很可能没有设置 `SERPAPI_API_KEY`，可在 `.env` 中补充或忽略搜索相关功能。

3. **输出看起来总是“空话”？**  
   - 检查是否未设置 `OPENAI_API_KEY`，此时使用的是 `DummyLLM` 占位输出，仅用于调试；配置真实 Key 后重新运行。

4. **Embeddings 初始化失败或 404？**  
   - 部分兼容接口（如 DeepSeek）暂不支持 embeddings，本项目会自动切换到 `DummyEmbeddings` 与简易向量检索，不影响主流程。

5. **bitsandbytes 安装失败？**  
   - 不使用 `--qlora` 即可，只跑普通 LoRA；或在支持的 GPU/驱动环境中安装 `bitsandbytes`。

6. **HTML 报告没有生成？**  
   - 仅 demo / eval / 长文本模式会生成 HTML 报告，`synthesize` 与 `distill` 仅输出 JSON 或 JSONL 文件。

7. **Agent B 的评分 JSON 解析失败？**  
   - 可能是模型未严格遵循 JSON 格式提示；可以提高模型质量、增加约束提示，或在后处理时增加更鲁棒的解析方式。

8. **Diff 太长或不易阅读？**  
   - 内部对 Diff 有 400 行的截断逻辑，可根据需要在 `DualAgentAcademicSystem._compute_diff` 中调整。

---
## 14. 后续扩展建议
- 在 `_plan_and_act()` 中新增领域检索工具（如论文数据库 API），增强事实依据；
- 扩展 Agent B 的评分维度与诊断模板，例如增加“清晰度”“严谨引用”“实验完整性”等；
- 引入并行分段处理与跨段全局精修，进一步提升长文档整体一致性；
- 将目前的指标与日志输出接入可视化面板（如 Streamlit / Gradio）构建交互 Demo UI。

本 README 与 `multi_agent_nlp_project.py` / `multi_agent_nlp_project.ipynb` 已按当前代码同步整理，可作为理解和使用整个“学术表达优化助手”项目的主参考文档。
