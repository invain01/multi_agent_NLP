import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import json
import argparse
import re
import requests
import sys
from pathlib import Path  # 重新添加 Path 导入以修复类型引用

# 在低依赖环境下回退 PromptTemplate 与 StrOutputParser
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    _PROMPT_AVAILABLE = True
except Exception as _e:
    print(f"⚠️ langchain_core prompts/output_parsers 导入失败({_e}); 使用轻量 stub")
    _PROMPT_AVAILABLE = False


    class PromptTemplate:
        def __init__(self, template: str):
            self.template = template

        @classmethod
        def from_template(cls, t: str):
            return cls(t)

        def format(self, **kwargs):
            return self.template.format(**kwargs)

        def __or__(self, other):  # 链接操作兼容
            return other


    class StrOutputParser:
        def __or__(self, other):
            return other

        def invoke(self, x):  # 保留接口
            return x

# 移除重复的 Path 导入，必要时再局部导入

# 添加项目根目录到路径以便导入 metrics 模块
sys.path.insert(0, os.path.dirname(__file__))

try:
    from metrics import AcademicMetrics
except ImportError:
    print("⚠️ metrics 模块导入失败，将使用基础指标计算占位逻辑")
    AcademicMetrics = None

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
IS_DEEPSEEK = "deepseek.com" in (OPENAI_BASE_URL or "").lower()
# Normalize DeepSeek model naming per official docs
if IS_DEEPSEEK:
    lm_lower = (LLM_MODEL or "").lower()
    if lm_lower not in ("deepseek-chat", "deepseek-reasoner"):
        new_model = "deepseek-reasoner" if ("reason" in lm_lower or "think" in lm_lower) else "deepseek-chat"
        print(f"ℹ️ 检测到 DeepSeek 且模型名 '{LLM_MODEL}' 非官方推荐，已自动规范为 '{new_model}'。")
        LLM_MODEL = new_model


class DummyLLM:
    """Fallback LLM used when API keys are missing; mimics .invoke interface and is Runnable-compatible."""

    def __init__(self):
        self.model_name = "dummy-llm"

    def invoke(self, prompt: Dict | str):
        if isinstance(prompt, dict):
            return f"[DummyLLM response for keys: {list(prompt.keys())}]"
        return "[DummyLLM generic response]"

    def __call__(self, prompt: Dict | str):  # 使其可调用, 便于 LangChain 将其视为 Runnable
        return self.invoke(prompt)

    def __or__(self, other):  # 在链式操作中返回自身, 让 LangChain 对 DummyLLM 执行调用
        return self


class HTTPFallbackChat:
    """
直接使用 OpenAI 兼容接口的简单回退客户端。满足 .invoke(dict|str) 接口。
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # 兼容有/无 /v1 前缀
        if self.base_url.endswith('/v1'):
            self.endpoint = f"{self.base_url}/chat/completions"
        else:
            self.endpoint = f"{self.base_url}/v1/chat/completions"

    def invoke(self, prompt: Dict | str):
        if isinstance(prompt, dict):
            # 将字典内容拼合为 user 消 Messages
            user_content = '\n'.join(f"{k}: {v}" for k, v in prompt.items())
        else:
            user_content = str(prompt)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an academic writing optimization assistant."},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                return f"[HTTPFallbackChat Error {resp.status_code}: {resp.text[:200]}]"
            data = resp.json()
            # OpenAI 兼容格式：choices[0].message.content
            return data.get("choices", [{}])[0].get("message", {}).get("content", "[No content]")
        except Exception as e:
            return f"[HTTPFallbackChat Exception: {e}]"

    def __call__(self, prompt: Dict | str):  # 增加直接调用支持
        return self.invoke(prompt)

    def __or__(self, other):
        return other


def init_llm():
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY missing; using DummyLLM.")
        return DummyLLM()
    # 优先尝试 langchain ChatOpenAI
    try:
        from langchain_openai import ChatOpenAI  # lazy import
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=(lambda: OPENAI_API_KEY),
            base_url=OPENAI_BASE_URL,
        )
        print(f"✅ Primary LLM ({LLM_MODEL}) initialized via {OPENAI_BASE_URL}.")
        return llm
    except Exception as e:
        print(f"❌ Failed to init ChatOpenAI: {e}; attempting HTTP fallback.")
        try:
            fallback_llm = HTTPFallbackChat(OPENAI_BASE_URL, OPENAI_API_KEY, LLM_MODEL)
            # 简单探针调用确认可用
            probe = fallback_llm.invoke("probe")
            if probe.startswith("[HTTPFallbackChat Error") or probe.startswith("[HTTPFallbackChat Exception"):
                print("⚠️ HTTP fallback probe failed; using DummyLLM.")
                return DummyLLM()
            print(f"✅ HTTP Fallback LLM ({LLM_MODEL}) ready via {OPENAI_BASE_URL}.")
            return fallback_llm
        except Exception as e2:
            print(f"❌ Fallback initialization failed: {e2}; using DummyLLM.")
            return DummyLLM()


llm = init_llm()

# LangChain / 工具层软回退：若依赖缺失则使用轻量占位实现保持接口兼容
try:
    from langchain_core.tools import Tool  # type: ignore
    from langchain_experimental.utilities import PythonREPL  # type: ignore
    from langchain_community.utilities import SerpAPIWrapper  # type: ignore

    _LC_AVAILABLE = True
except Exception as e:
    print(f"⚠️ LangChain 相关依赖缺失或导入失败({e}); 使用轻量 stub 替代，功能受限但流程可继续。")
    _LC_AVAILABLE = False


    class Tool:  # minimal stub
        def __init__(self, name: str, func, description: str = ""):
            self.name = name
            self.func = func
            self.description = description

        def run(self, *args, **kwargs):
            return self.func(*args, **kwargs)


    class PythonREPL:  # stub REPL
        def run(self, code: str) -> str:
            try:
                local_env = {}
                exec(code, {}, local_env)
                return str(local_env)[:500]
            except Exception as e2:
                return f"[PythonREPL stub 异常: {e2}]"


    class SerpAPIWrapper:  # stub search
        def __init__(self, *_, **__):
            pass

        def run(self, query: str) -> str:
            return f"[SerpAPI stub 未配置，无法真实检索: {query}]"

TOOLS: List[Tool] = []  # add explicit type for clarity
if SERPAPI_API_KEY:
    search_wrapper = SerpAPIWrapper(search_engine="google", serpapi_api_key=SERPAPI_API_KEY)
    search_tool = Tool(
        name="网络搜索",
        func=search_wrapper.run,
        description="实时信息查询：输入搜索关键词"
    )
    TOOLS.append(search_tool)
else:
    def _search_stub(q: str) -> str:
        return f"[SerpAPI 未配置，无法执行搜索: {q}]"


    search_tool = Tool(name="网络搜索", func=_search_stub, description="SerpAPI 未配置占位工具")
    TOOLS.append(search_tool)

python_repl = PythonREPL()
python_repl_tool = Tool(
    name="Python REPL",
    func=python_repl.run,
    description="执行格式正确的 Python 代码"
)
TOOLS.extend([
    python_repl_tool,
    # 简化文件读写工具，避免依赖未定义的 read_file/write_file
    Tool(
        name="读取文件",
        func=lambda fn: open(fn, "r", encoding="utf-8").read(),
        description="读取指定文本文件的全部内容，输入为文件路径"
    ),
    Tool(
        name="写入文件",
        func=lambda arg: (
            (lambda filename, content: (open(filename, "w", encoding="utf-8").write(content), "写入完成")[1])
        )(*arg.split(",", 1)),
        description="写入文件内容。输入格式: 文件名,内容（用英文逗号分隔，内容中若有逗号视为正文的一部分）"
    ),
])

print(f"🔧 已加载 {len(TOOLS)} 个工具")

EMBED_DIM = 1536  # embedding vector size defined early for stub usage
USE_FAISS = True
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
    import faiss
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings as LCEmbeddings
except ImportError as e:
    print(f"⚠️ 向量存储依赖缺失: {e}. 使用简易内存检索代替 FAISS。")
    USE_FAISS = False
    FAISS = None  # type: ignore
    InMemoryDocstore = None  # type: ignore
    faiss = None  # type: ignore
    try:
        from langchain_core.documents import Document  # may still be available
    except Exception:
        class Document:  # minimal fallback
            def __init__(self, page_content: str, metadata: Optional[Dict] = None):
                self.page_content = page_content
                self.metadata = metadata or {}


    class LCEmbeddings:  # type: ignore
        _dim = EMBED_DIM if 'EMBED_DIM' in globals() else 1536

        def embed_query(self, x: str):
            return [0.0] * self._dim

        def embed_documents(self, xs: List[str]):
            return [[0.0] * self._dim for _ in xs]
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "text-embedding-3-small")


class DummyEmbeddings:
    def embed_query(self, t: str):
        return [0.0] * EMBED_DIM

    def embed_documents(self, docs: List[str]):
        return [[0.0] * EMBED_DIM for _ in docs]

    def __call__(self, t: str):
        return self.embed_query(t)


if OPENAI_API_KEY:
    try:
        if IS_DEEPSEEK:
            # DeepSeek 的 OpenAI 兼容接口通常不提供 embeddings；直接使用占位向量避免 404
            print("ℹ️ 检测到 DeepSeek base_url，跳过 OpenAIEmbeddings，使用占位嵌入以启用记忆功能。")
            embeddings_model = DummyEmbeddings()
        else:
            from langchain_openai import OpenAIEmbeddings  # lazy import

            embeddings_model = OpenAIEmbeddings(
                model=EMBED_MODEL_NAME,
                api_key=(lambda: OPENAI_API_KEY),
                base_url=OPENAI_BASE_URL
            )
    except Exception as e:
        print(f"❌ Embeddings 初始化失败: {e}; 使用占位 embedding 函数。")
        embeddings_model = DummyEmbeddings()
else:
    embeddings_model = DummyEmbeddings()
    print("⚠️ OPENAI_API_KEY 缺失，向量嵌入使用占位向量。")

_DEF_WORD_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


def _simple_tokenize(text: str) -> List[str]:
    return _DEF_WORD_RE.findall(text or "")


if USE_FAISS:
    class EmbeddingAdapter(LCEmbeddings):
        """Embeddings adapter implementing LangChain Embeddings interface to silence deprecation warnings."""

        def __init__(self, base):
            self.base = base

        def embed_query(self, x: str) -> List[float]:
            try:
                return self.base.embed_query(x)
            except Exception:
                return [0.0] * EMBED_DIM

        def embed_documents(self, xs: List[str]) -> List[List[float]]:
            try:
                return self.base.embed_documents(xs)
            except Exception:
                return [[0.0] * EMBED_DIM for _ in xs]


    adapter = EmbeddingAdapter(embeddings_model)
    index = faiss.IndexFlatL2(EMBED_DIM)
    vectorstore = FAISS(adapter, index, InMemoryDocstore({}), {})
    print("🧠 向量数据库(FAISS)初始化完成")
else:
    class SimpleVectorStore:
        def __init__(self):
            self.docs: List[Document] = []

        def add_documents(self, docs: List[Document]):
            self.docs.extend(docs)

        def similarity_search(self, query: str, k: int = 3) -> List[Document]:
            q_tokens = set(_simple_tokenize(query))

            def score(doc: Document):
                d_tokens = set(_simple_tokenize(doc.page_content))
                if not q_tokens or not d_tokens:
                    return 0.0
                return len(q_tokens & d_tokens) / len(q_tokens | d_tokens)

            ranked = sorted(self.docs, key=score, reverse=True)
            return ranked[:k]


    vectorstore = SimpleVectorStore()
    print("🧠 向量数据���简化版初始化完成(无FAISS)")


class MemoryManager:
    def __init__(self, vs, namespace: str = "global"):
        self.vs = vs
        self.namespace = namespace
        self._counter = 0

    def add_memory(self, text: str, metadata: Optional[Dict] = None):
        try:
            meta = metadata or {}
            meta.update({"namespace": self.namespace, "ts": datetime.now().isoformat()})
            doc = Document(page_content=text, metadata=meta)
            self.vs.add_documents([doc])
            self._counter += 1
        except Exception as e:
            print(f"⚠️ 写入记忆失败: {e}")

    def recall(self, query: str, k: int = 3) -> List[str]:
        try:
            res = self.vs.similarity_search(query, k=k)
            return [d.page_content for d in res]
        except Exception as e:
            print(f"⚠️ 读取记忆失败: {e}")
            return []


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class DualAgentAcademicSystem:
    def __init__(self, llm, tools, vectorstore, enable_tools: bool = True, enable_memory: bool = True,
                 agent_a_llm=None, agent_b_llm=None):
        """Dual-agent academic system.

        Args:
            llm: Default LLM used for both agents when agent_a_llm/agent_b_llm are not provided.
            tools: LangChain Tool list.
            vectorstore: Vector store or SimpleVectorStore.
            enable_tools: Whether to use external tools.
            enable_memory: Whether to use vectorstore-backed long-term memory.
            agent_a_llm: Optional dedicated LLM for Agent A (optimizer), e.g. local Qwen.
            agent_b_llm: Optional dedicated LLM for Agent B (reviewer), e.g. remote DeepSeek.
        """
        self.llm = llm
        self.agent_a_llm = agent_a_llm or llm
        self.agent_b_llm = agent_b_llm or llm
        self.tools_enabled = enable_tools
        self.memory_enabled = enable_memory
        self.tools = {t.name: t for t in tools}
        self.vectorstore = vectorstore
        self.memory = MemoryManager(vectorstore) if enable_memory else None
        self.collaboration_log: List[Dict] = []
        self._setup_agents()

    def _setup_agents(self):
        self.agent_a_template = PromptTemplate.from_template(
            """你是Agent A - 学术表达优化专家。\n轮次: 第{round_num}轮\n用户需求: {user_requirements}\n上一轮评分(若有): {last_scores}\n长程记忆检索片段:\n{memory_snippets}\n工具观察:\n{tool_observations}\n待优化文本:\n{text_to_optimize}\n{previous_feedback}\n请输出:\n**优化版本：**\n[优化后的完整文本]\n\n**修改说明：**\n[说明本轮修改要点，尤其针对评审提出的高优先级问题]"""
        )
        self.agent_b_template = PromptTemplate.from_template(
            """你是Agent B - 学术评审与对抗质询专家。\n轮次: 第{round_num}轮\n用户需求: {user_requirements}\n优化文本:\n{optimized_text}\n请评审并输出(严格包含以下板块与数值)：\n**本轮改进评价：**\n[总体评价]\n\n**评分(请使用JSON格式)**\n{{"quality": <1-10>, "rigor": <1-10>, "logic": <1-10>, "novelty": <1-10>, "priority_issues": <描述>}}\n\n**剩余主要问题：**\n[...]\n\n**下轮重点建议：**\n1. [...]\n2. [...]\n\n**改进优先级：**\n[高/中/低 分层列出]"""
        )
        # use dedicated LLMs for each agent
        self.agent_a_chain = self.agent_a_template | self.agent_a_llm | StrOutputParser()
        self.agent_b_chain = self.agent_b_template | self.agent_b_llm | StrOutputParser()

    @staticmethod
    def _extract_section(text: str, start_token: str, end_token: str) -> str:
        lines = text.split('\n')
        collecting = False
        buf = []
        for l in lines:
            if start_token in l:
                collecting = True
                continue
            if collecting and end_token in l:
                break
            if collecting:
                buf.append(l)
        return '\n'.join(buf).strip()

    def _compute_diff(self, prev: str, current: str) -> str:
        """Unified diff between previous and current text (truncated)."""
        if prev is None:
            return '(首轮无diff)'
        import difflib as _df
        diff_lines = _df.unified_diff(prev.splitlines(), current.splitlines(), lineterm='')
        collected = []
        for i, line in enumerate(diff_lines):
            if i > 400:
                collected.append('... <diff truncated>')
                break
            collected.append(line)
        return '\n'.join(collected) if collected else '(无变化)'

    def _parse_scores(self, feedback: str) -> Dict[str, float]:
        import json as _json
        import re as _re
        # 修正正则：去除冗余的转义 \}
        # _parse_scores 中原模式 r"\{\s*\"quality\".*?}" -> 保留不转义的 }
        m = _re.search(r"\{\s*\"quality\".*?}", feedback, flags=_re.S)
        if not m:
            return {}
        blob = m.group(0)
        try:
            data = _json.loads(blob)
            for k in ["quality", "rigor", "logic", "novelty"]:
                if k in data:
                    data[k] = float(data[k])
            return data
        except Exception:
            return {}

    def _plan_and_act(self, text: str, requirements: List[str]) -> str:
        if not self.tools_enabled:
            return "(工具已禁用)"
        observations = []
        joined_req = ' '.join(requirements).lower()
        if any(kw in joined_req for kw in ["search", "检索", "事实", "最新", "引用"]):
            m = re.findall(r'"([^\"]+)"', text)
            query = m[-1] if m else text.split('。')[-1].strip() or text
            try:
                obs = self.tools["网络搜索"].run(query)
            except Exception as e:
                obs = f"[搜索异常: {e}]"
            observations.append(f"搜索[{query}] -> {obs[:300]}")
        code_blocks = re.findall(r'python:\s*```python\n([\s\S]*?)```', text, flags=re.IGNORECASE)
        for code in code_blocks[:1]:
            try:
                out = self.tools["Python REPL"].run(code)
            except Exception as e:
                out = f"[代码执行异常: {e}]"
            observations.append(f"执行Python -> 输出: {str(out)[:200]}")
        return "\n".join(observations) if observations else "(无)"

    def collaborate(self, user_text: str, user_requirements: List[str], language: str = "中文", rounds: int = 3) -> \
    Tuple[str, List[Dict]]:
        self.collaboration_log = [{"round": 0, "user_input": user_text, "requirements": user_requirements,
                                   "timestamp": datetime.now().isoformat()}]
        current_text = user_text
        previous_feedback = ""
        last_scores = {}
        if self.memory_enabled:
            self.memory.add_memory(user_text, {"type": "user_input"})
        for r in range(1, rounds + 1):
            mem_snippets = []
            if self.memory_enabled:
                mem_snippets = self.memory.recall(current_text, k=3)
            tool_obs = self._plan_and_act(current_text, user_requirements)
            a_input = {
                "round_num": r,
                "text_to_optimize": current_text,
                "user_requirements": ', '.join(user_requirements),
                "previous_feedback": previous_feedback,
                "memory_snippets": '\n'.join(mem_snippets) if mem_snippets else "(无)",
                "tool_observations": tool_obs,
                "last_scores": last_scores if last_scores else "(无)"
            }
            a_resp = self.agent_a_chain.invoke(a_input)
            optimized_text = self._extract_section(a_resp, "**优化版本：**", "**修改说明：**") or current_text
            b_input = {
                "round_num": r,
                "optimized_text": optimized_text,
                "user_requirements": ', '.join(user_requirements)
            }
            b_resp = self.agent_b_chain.invoke(b_input)
            last_scores = self._parse_scores(b_resp)
            diff_str = self._compute_diff(current_text, optimized_text)
            if self.memory_enabled:
                self.memory.add_memory(optimized_text, {"type": "optimized_text", "round": r})
                self.memory.add_memory(b_resp, {"type": "feedback", "round": r})
            self.collaboration_log.append({
                "round": r,
                "agent_a_response": a_resp,
                "optimized_text": optimized_text,
                "agent_b_feedback": b_resp,
                "scores": last_scores,
                "tool_observations": tool_obs,
                "diff": diff_str,
                "timestamp": datetime.now().isoformat()
            })
            previous_feedback = b_resp
            current_text = optimized_text
            print(f"✅ Round {r} 完成 | 评分: {last_scores if last_scores else '{}'}")
            time.sleep(0.15)
        return current_text, self.collaboration_log

    def synthesize_dataset(self, seeds: List[str], requirements: List[str], rounds: int = 3,
                           out_path: Optional["Path"] = None) -> "Path":
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        if out_path is None:
            out_path = data_dir / f"synth_academic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for i, seed in enumerate(seeds):
                final_text, log = self.collaborate(seed, requirements, rounds=rounds)
                record = {
                    "id": f"case_{i}",
                    "input": seed,
                    "requirements": requirements,
                    "final": final_text,
                    "log": log,
                    "created_at": datetime.now().isoformat(),
                    "teacher_signal": log[-1].get("optimized_text", final_text),
                    "scores": log[-1].get("scores", {})
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        print(f"📀 合成数据已写入: {out_path} (共 {count} 条)")
        return out_path

    @staticmethod
    def _tokenize_zh(text: str) -> List[str]:
        words = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text)
        return words

    def _readability_proxy(self, text: str) -> float:
        sentences = [s for s in re.split(r'[。.!?]\s*', text) if s.strip()]
        if not sentences:
            return 0.0
        avg_len = sum(len(s) for s in sentences) / len(sentences)
        # Normalize (heuristic) shorter sentences -> higher readability (invert)
        return round(1 / (1 + avg_len / 50), 4)

    def _coherence_proxy(self, text: str) -> float:
        sentences = [s for s in re.split(r'[。.!?]\s*', text) if s.strip()]
        if len(sentences) < 2:
            return 0.0

        def tokens(s):
            return set(self._tokenize_zh(s))

        overlaps = []
        for a, b in zip(sentences[:-1], sentences[1:]):
            ta, tb = tokens(a), tokens(b)
            if ta and tb:
                overlaps.append(len(ta & tb) / len(ta | tb))
        return round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0

    def evaluate(self, cases: List[Tuple[str, List[str]]], rounds: int = 2) -> Dict:
        results = []
        for idx, (text, reqs) in enumerate(cases):
            final_text, log = self.collaborate(text, reqs, rounds=rounds)
            w0 = self._tokenize_zh(text)
            w1 = self._tokenize_zh(final_text)
            len_gain = (len(w1) - len(w0)) / max(1, len(w0))
            ttr0 = len(set(w0)) / max(1, len(w0))
            ttr1 = len(set(w1)) / max(1, len(w1))
            from collections import Counter
            c0 = Counter(w0)
            c1 = Counter(w1)
            rep0 = sum(x for _, x in c0.most_common(5)) / max(1, len(w0))
            rep1 = sum(x for _, x in c1.most_common(5)) / max(1, len(w1))
            readability_gain = self._readability_proxy(final_text) - self._readability_proxy(text)
            coherence_gain = self._coherence_proxy(final_text) - self._coherence_proxy(text)

            def _sentence_lengths(t: str):
                sents = [s for s in re.split(r'[。.!?]\s*', t) if s.strip()]
                return [len(s) for s in sents] if sents else []

            import statistics
            var0 = statistics.pvariance(_sentence_lengths(text)) if _sentence_lengths(text) else 0.0
            var1 = statistics.pvariance(_sentence_lengths(final_text)) if _sentence_lengths(final_text) else 0.0
            var_delta = round(var0 - var1, 3)

            def _bigram_rep(t: str):
                toks = w1 if t == final_text else w0
                bigrams = [tuple(toks[i:i + 2]) for i in range(len(toks) - 1)]
                bc = Counter(bigrams)
                total = len(bigrams) or 1
                top = sum(v for _, v in bc.most_common(5))
                return top / total

            bigram_delta = round(_bigram_rep(text) - _bigram_rep(final_text), 3)
            last_scores = log[-1].get("scores", {}) if log else {}

            # ============ 使用新的学术指标系统 ============
            advanced_metrics = {}
            if AcademicMetrics:
                try:
                    original_eval = AcademicMetrics.overall_quality_score(text)
                    optimized_eval = AcademicMetrics.overall_quality_score(final_text)
                    # 结构健壮性保护
                    orig_scores = (original_eval or {}).get('scores', {})
                    opt_scores = (optimized_eval or {}).get('scores', {})
                    comparison = AcademicMetrics.compare_improvements(text, final_text)
                    advanced_metrics = {
                        'original_overall_score': float((original_eval or {}).get('overall_score', 0.0)),
                        'optimized_overall_score': float((optimized_eval or {}).get('overall_score', 0.0)),
                        'academic_formality_improvement': round(
                            float(opt_scores.get('academic_formality', 0.0)) - float(
                                orig_scores.get('academic_formality', 0.0)), 4),
                        'citation_completeness_improvement': round(
                            float(opt_scores.get('citation_completeness', 0.0)) - float(
                                orig_scores.get('citation_completeness', 0.0)), 4),
                        'novelty_improvement': round(
                            float(opt_scores.get('novelty', 0.0)) - float(orig_scores.get('novelty', 0.0)), 4),
                        'language_fluency_improvement': round(float(opt_scores.get('language_fluency', 0.0)) - float(
                            orig_scores.get('language_fluency', 0.0)), 4),
                        'sentence_balance_improvement': round(float(opt_scores.get('sentence_balance', 0.0)) - float(
                            orig_scores.get('sentence_balance', 0.0)), 4),
                        'argumentation_improvement': round(
                            float(opt_scores.get('argumentation', 0.0)) - float(orig_scores.get('argumentation', 0.0)),
                            4),
                        'expression_diversity_improvement': round(
                            float(opt_scores.get('expression_diversity', 0.0)) - float(
                                orig_scores.get('expression_diversity', 0.0)), 4),
                        'structure_completeness_improvement': round(
                            float(opt_scores.get('structure_completeness', 0.0)) - float(
                                orig_scores.get('structure_completeness', 0.0)), 4),
                        'tense_consistency_improvement': round(float(opt_scores.get('tense_consistency', 0.0)) - float(
                            orig_scores.get('tense_consistency', 0.0)), 4),
                    }
                except Exception as e:
                    print(f"⚠️ 计算高级指标失败: {e}")

            results.append({
                "id": idx,
                "len_gain": round(len_gain, 3),
                "ttr_gain": round(ttr1 - ttr0, 3),
                "repetition_delta": round(rep0 - rep1, 3),
                "readability_gain": round(readability_gain, 3),
                "coherence_gain": round(coherence_gain, 3),
                "sent_var_delta": var_delta,
                "bigram_rep_delta": round(bigram_delta, 3),
                "orig_len": len(w0),
                "final_len": len(w1),
                "scores": last_scores,
                "advanced_metrics": advanced_metrics
            })
        # aggregate
        if results:
            avg = {
                "len_gain_avg": round(sum(r["len_gain"] for r in results) / len(results), 3),
                "ttr_gain_avg": round(sum(r["ttr_gain"] for r in results) / len(results), 3),
                "repetition_delta_avg": round(sum(r["repetition_delta"] for r in results) / len(results), 3),
                "readability_gain_avg": round(sum(r["readability_gain"] for r in results) / len(results), 3),
                "coherence_gain_avg": round(sum(r["coherence_gain"] for r in results) / len(results), 3),
                "sent_var_delta_avg": round(sum(r["sent_var_delta"] for r in results) / len(results), 3),
                "bigram_rep_delta_avg": round(sum(r["bigram_rep_delta"] for r in results) / len(results), 3),
                "quality_avg": round(sum(r.get("scores", {}).get("quality", 0) for r in results) / len(results), 3),
                "rigor_avg": round(sum(r.get("scores", {}).get("rigor", 0) for r in results) / len(results), 3),
                "logic_avg": round(sum(r.get("scores", {}).get("logic", 0) for r in results) / len(results), 3),
                "novelty_avg": round(sum(r.get("scores", {}).get("novelty", 0) for r in results) / len(results), 3),
                "n": len(results)
            }

            # 添加高级指标平均值
            if results[0].get("advanced_metrics"):
                adv_avg = {}
                for metric in results[0]["advanced_metrics"].keys():
                    adv_avg[f"{metric}_avg"] = round(
                        sum(r.get("advanced_metrics", {}).get(metric, 0) for r in results) / len(results), 4)
                avg.update(adv_avg)
        else:
            avg = {"len_gain_avg": 0, "ttr_gain_avg": 0, "repetition_delta_avg": 0, "readability_gain_avg": 0,
                   "coherence_gain_avg": 0, "sent_var_delta_avg": 0, "bigram_rep_delta_avg": 0, "quality_avg": 0,
                   "rigor_avg": 0, "logic_avg": 0, "novelty_avg": 0, "n": 0}
        report = {"summary": avg, "cases": results}
        print("📈 评估汇总:", json.dumps(report["summary"], ensure_ascii=False))
        return report

    def prepare_distillation_pairs(self, jsonl_path: "Path", out_path: "Path") -> "Path":
        pairs = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for ln in f:
                if not ln.strip():
                    continue
                obj = json.loads(ln)
                instr = f"优化以下学术段落，满足需求: {', '.join(obj.get('requirements', []))}\n原文: {obj.get('input', '')}"
                target = obj.get('teacher_signal', obj.get('final', ''))
                scores = obj.get('scores', {})
                pairs.append({"instruction": instr, "output": target, "scores": scores})
        with open(out_path, 'w', encoding='utf-8') as w:
            for p in pairs:
                w.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"🧪 蒸馏数据已生成: {out_path} 共 {len(pairs)} 条")
        return out_path


# Helper to build a hybrid system where Agent A uses a local HF Qwen student model
# and Agent B uses the existing remote teacher llm.
try:
    from hf_student_llm import HFChatLLM as _HFStudentLLM
except Exception:
    _HFStudentLLM = None


def build_hybrid_dual_agent_system(base_model: str | None = None,
                                   lora_dir: str | None = None,
                                   max_new_tokens: int | None = None,
                                   device: str | None = None,
                                   torch_dtype: str | None = None,
                                   device_map: str | None = None,
                                   load_in_4bit: bool | None = None) -> DualAgentAcademicSystem:
    """Create a DualAgentAcademicSystem with local HF student model for Agent A and remote LLM for Agent B.
    Added detection of FORCE_STUDENT_STUB=1 测试/CI模式.
    支持延迟加载和单例模式以节省内存.
    """
    if _HFStudentLLM is None:
        print("⚠️ HF student LLM not available (transformers/peft missing or import error); using single LLM mode.")
        return DualAgentAcademicSystem(llm, TOOLS, vectorstore)

    # Resolve overrides with env fallback
    env_base = os.getenv("STUDENT_BASE_MODEL", "Qwen/Qwen1.5-1.8B-Chat")
    env_lora = os.getenv("STUDENT_LORA_DIR") or ""
    env_max_new = int(os.getenv("STUDENT_MAX_NEW_TOKENS", "512"))

    base_model = base_model or env_base
    lora_dir = lora_dir if lora_dir is not None else env_lora
    max_new = max_new_tokens or env_max_new

    if lora_dir and not Path(lora_dir).exists():
        print(f"⚠️ STUDENT_LORA_DIR not found ({lora_dir}); will load base model without LoRA.")
        lora_dir = ""  # 继续但不使用LoRA

    stub_mode = os.getenv("FORCE_STUDENT_STUB") == "1"
    if stub_mode:
        print("🧪 FORCE_STUDENT_STUB=1 启用：学生模型使用轻量占位生成，不下载真实权重。")

    try:
        student_llm = _HFStudentLLM(base_model=base_model,
                                    lora_dir=lora_dir,
                                    max_new_tokens=max_new,
                                    device=device,
                                    torch_dtype=torch_dtype,
                                    device_map=device_map,
                                    load_in_4bit=load_in_4bit)
        print(f"🤖 Hybrid mode engaged | Agent A: {getattr(student_llm,'model_name', base_model)} (LoRA={'ON' if lora_dir else 'OFF'} stub={stub_mode}) | Agent B: {getattr(llm, 'model_name', 'remote-llm')}")
        return DualAgentAcademicSystem(llm, TOOLS, vectorstore, agent_a_llm=student_llm, agent_b_llm=llm)
    except Exception as e:
        print(f"⚠️ Failed to init student model '{base_model}': {e}; falling back to single LLM mode.")
        return DualAgentAcademicSystem(llm, TOOLS, vectorstore)


# 全局单例变量，支持延迟加载
_dual_agent_system_instance = None

def get_dual_agent_system(force_reload: bool = False) -> DualAgentAcademicSystem:
    """获取双 Agent 系统实例（单例模式 + 延迟加载）
    
    在Web应用中，这避免了在模块导入时就加载大型模型，节省内存。
    """
    global _dual_agent_system_instance
    if _dual_agent_system_instance is None or force_reload:
        print("⏳ 延迟加载双 Agent 系统...")
        _dual_agent_system_instance = build_hybrid_dual_agent_system()
        print("✅ 双 Agent 系统加载完成")
        if os.getenv("FORCE_STUDENT_STUB") == "1":
            print("🧪 当前运行于学生模型 STUB 模式：所有学生输出为占位生成，仅用于快速测试。")
    return _dual_agent_system_instance

# 保留向后兼容的接口，但不立即加载模型
# 用户可以直接调用 dual_agent_system，但实际上会延迟加载
class _LazyDualAgentSystemProxy:
    """Proxy 对象，延迟加载真实的 DualAgentAcademicSystem"""
    def __getattr__(self, name):
        # 当访问任何属性或方法时，才真正加载模型
        real_system = get_dual_agent_system()
        return getattr(real_system, name)

dual_agent_system = _LazyDualAgentSystemProxy()
print("🧠 双 Agent 系统代理已初始化（延迟加载模式，首次使用时才加载模型）")

# 修复：补全字符串引号
ENABLE_INTERACTIVE = os.getenv("ENABLE_INTERACTIVE", "0") == "1"
from html import escape as _html_escape


def generate_html_report(title: str, final_text: str, log: List[Dict], summary: Optional[Dict] = None) -> str:
    if 'round-box' in str(log[:1]):
        return ''

    def style():
        return '''<style>
body{font-family:Segoe UI,Arial,sans-serif;max-width:1200px;margin:32px auto;line-height:1.5}
pre{background:#fafafa;border:1px solid #eee;padding:8px;white-space:pre-wrap;overflow-x:auto}
table{border-collapse:collapse;width:100%;margin:10px 0}
td,th{border:1px solid #ccc;padding:6px 10px;font-size:13px;text-align:left}
th{background:#f0f0f0;font-weight:bold}
.score-badge{display:inline-block;padding:3px 8px;border-radius:4px;background:#004d7a;color:#fff;font-size:12px;margin-right:4px;margin-bottom:4px}
.metric-value{font-weight:bold;color:#2c3e50}
.improvement-positive{color:#27ae60;font-weight:bold}
.improvement-negative{color:#e74c3c;font-weight:bold}
.diff-add{background:#e6ffe6;padding:2px 4px}
.diff-del{background:#ffecec;color:#900;padding:2px 4px}
.round-box{border:1px solid #ddd;padding:12px;margin:12px 0;border-radius:6px;background:#fafafa}
.meta{color:#666;font-size:12px}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:15px;margin:15px 0}
.metric-card{border:1px solid #e0e0e0;padding:12px;border-radius:6px;background:white}
.metric-card h4{margin:0 0 8px 0;color:#2c3e50;font-size:14px}
.metric-card .value{font-size:20px;font-weight:bold;color:#3498db}
.metric-card .label{font-size:12px;color:#7f8c8d}
.chart-container{margin:15px 0;padding:15px;background:#f9f9f9;border-radius:6px}
details summary{cursor:pointer;font-weight:bold;padding:8px;background:#f0f0f0;margin:8px 0;border-radius:4px}
.progress-bar{width:100%;height:20px;background:#ecf0f1;border-radius:4px;overflow:hidden;margin:5px 0}
.progress-fill{height:100%;background:linear-gradient(to right,#3498db,#2980b9);transition:width 0.3s}
.warning{background:#fff3cd;border-left:4px solid #ffc107;padding:10px;margin:10px 0}
.success{background:#d4edda;border-left:4px solid #28a745;padding:10px;margin:10px 0}
</style>'''

    def render_scores(scores: Dict[str, float]) -> str:
        if not scores:
            return '<span class="meta">无评分</span>'
        return ' '.join(
            f'<span class="score-badge">{k}:{v:.1f}</span>' for k, v in scores.items() if isinstance(v, (int, float)))

    def render_metric_cards(metrics: Dict) -> str:
        """生成指标卡片网格"""
        if not metrics:
            return ''

        html = '<div class="metric-grid">'
        for key, value in metrics.items():
            if key.endswith('_avg') or key.endswith('_improvement'):
                # 提取易读的标签名
                label = key.replace('_avg', '').replace('_improvement', '').replace('_', ' ').title()

                # 确定颜色和方向
                if isinstance(value, (int, float)):
                    if value > 0:
                        color_class = 'improvement-positive'
                        symbol = '↑'
                    elif value < 0:
                        color_class = 'improvement-negative'
                        symbol = '↓'
                    else:
                        color_class = ''
                        symbol = '='

                    html += f'''
                    <div class="metric-card">
                        <h4>{label}</h4>
                        <div class="value {color_class}">{symbol} {value:.4f}</div>
                        <div class="label">改进幅度</div>
                    </div>
                    '''
        html += '</div>'
        return html

    def render_advanced_metrics(advanced_metrics: Dict) -> str:
        """渲染高级学术指标"""
        if not advanced_metrics:
            return ''

        html = '<h3>高级学术指标</h3>'
        html += '<div class="metric-grid">'

        # 原始和优化后的总体评分
        if 'original_overall_score' in advanced_metrics and 'optimized_overall_score' in advanced_metrics:
            orig = advanced_metrics['original_overall_score']
            opt = advanced_metrics['optimized_overall_score']
            improvement = opt - orig

            html += f'''
            <div class="metric-card">
                <h4>总体质量评分</h4>
                <div style="font-size:12px;margin-bottom:8px">
                    优化前: <span class="metric-value">{orig:.4f}</span><br/>
                    优化后: <span class="metric-value">{opt:.4f}</span>
                </div>
                <div class="improvement-{('positive' if improvement > 0 else 'negative')}">
                    {improvement:+.4f}
                </div>
            </div>
            '''

        # 各维度改进
        improvements = {k: v for k, v in advanced_metrics.items() if '_improvement' in k and k != 'overall_improvement'}
        for metric, improvement in sorted(improvements.items())[:6]:  # 显示前6个
            label = metric.replace('_improvement', '').replace('_', ' ').title()
            color_class = 'improvement-positive' if improvement > 0 else 'improvement-negative'

            # 创建进度条
            progress = max(0, min((improvement + 0.5) / 1.0 * 100, 100))  # 标准化到0-100%

            html += f'''
            <div class="metric-card">
                <h4>{label}</h4>
                <div class="progress-bar">
                    <div class="progress-fill" style="width:{progress}%"></div>
                </div>
                <div class="{color_class}">{improvement:+.4f}</div>
            </div>
            '''

        html += '</div>'
        return html

    def color_diff(diff_text: str) -> str:
        lines = []
        for ln in diff_text.splitlines():
            if ln.startswith('+') and not ln.startswith('+++'):
                lines.append(f'<div class="diff-add">{_html_escape(ln)}</div>')
            elif ln.startswith('-') and not ln.startswith('---'):
                lines.append(f'<div class="diff-del">{_html_escape(ln)}</div>')
            else:
                lines.append(f'<div>{_html_escape(ln)}</div>')
        return '\n'.join(lines)

    parts = [f'<html><head><meta charset="utf-8"><title>{_html_escape(title)}</title>{style()}</head><body>']
    parts.append(f'<h1>📊 {_html_escape(title)}</h1>')
    parts.append(f'<div class="meta">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>')

    if summary:
        parts.append('<h2>📈 指标汇总</h2>')

        # 显示基础指标表格
        parts.append('<h3>基础文本指标</h3>')
        parts.append('<table><tr>')
        basic_metrics = {
            'len_gain_avg': '长度变化',
            'ttr_gain_avg': '词汇多样性提升',
            'repetition_delta_avg': '重复度下降',
            'readability_gain_avg': '可读性提升',
            'coherence_gain_avg': '连贯性提升'
        }
        for key in basic_metrics:
            parts.append(f'<th>{basic_metrics[key]}</th>')
        parts.append('</tr><tr>')
        for key in basic_metrics:
            val = summary.get(key, 0)
            color = 'improvement-positive' if val > 0 else 'improvement-negative' if val < 0 else ''
            parts.append(f'<td class="{color}">{val:.4f}</td>')
        parts.append('</tr></table>')

        # 显示Agent评分
        parts.append('<h3>Agent B 评分汇总</h3>')
        parts.append('<table><tr>')
        agent_metrics = {
            'quality_avg': '质量',
            'rigor_avg': '严谨性',
            'logic_avg': '逻辑性',
            'novelty_avg': '新颖性'
        }
        for key in agent_metrics:
            parts.append(f'<th>{agent_metrics[key]}</th>')
        parts.append('</tr><tr>')
        for key in agent_metrics:
            val = summary.get(key, 0)
            parts.append(f'<td><span class="metric-value">{val:.2f}/10</span></td>')
        parts.append('</tr></table>')

        # 显示高级指标卡片
        adv_metrics = {k: v for k, v in summary.items() if 'improvement' in k}
        if adv_metrics:
            parts.append(render_metric_cards(adv_metrics))

    parts.append('<h2>📝 最终优化文本</h2><pre>' + _html_escape(final_text) + '</pre>')
    parts.append('<h2>📋 轮次详细日志</h2>')
    for entry in log[1:]:
        parts.append('<div class="round-box">')
        parts.append(f'<h3>Round {entry.get("round")}</h3>')
        parts.append(f'<div class="meta">时间: {entry.get("timestamp")}</div>')
        parts.append('<h4>✏️ 优化文本</h4><pre>' + _html_escape(entry.get('optimized_text', '')) + '</pre>')
        parts.append('<h4>📝 Agent B 反馈</h4><pre>' + _html_escape(entry.get('agent_b_feedback', '')) + '</pre>')
        parts.append('<h4>⭐ 评分</h4>' + render_scores(entry.get('scores', {})))
        parts.append('<details><summary>🔍 文本差异 (Diff)</summary>' + color_diff(entry.get('diff', '')) + '</details>')
        if entry.get('tool_observations') and entry.get('tool_observations') not in ('(无)', '(工具已禁用)'):
            parts.append('<details><summary>🔧 工具调用观察</summary><pre>' + _html_escape(
                entry.get('tool_observations', '')) + '</pre></details>')
        parts.append('</div>')
    parts.append('</body></html>')
    return '\n'.join(parts)


def parse_requirements(raw: Optional[str], fallback: List[str]) -> List[str]:
    if not raw:
        return fallback
    parts = re.split(r"[;,；]", raw)
    return [p.strip() for p in parts if p.strip()]


def load_seeds_from_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"⚠️ 种子文件不存在: {path}")
        return []
    return [l.strip() for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]


# ---------- Long text splitting & file optimization helpers ----------
def _split_long_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split long text into roughly chunk_size segments with sentence-aware boundaries.
    overlap: number of characters to prepend from previous chunk for continuity (ignored for first chunk)."""
    text = text.strip()
    if chunk_size <= 0:
        return [text]
    # Sentence tokenize (simple)
    sentences = re.split(r'([。.!?]\s*)', text)  # keep delimiters
    combined = []
    # Reconstruct with delimiters preserved
    buf = ''
    for i in range(0, len(sentences), 2):
        seg = sentences[i]
        delim = sentences[i + 1] if i + 1 < len(sentences) else ''
        piece = seg + delim
        if len(buf) + len(piece) <= chunk_size:
            buf += piece
        else:
            if buf:
                combined.append(buf)
            buf = piece
    if buf:
        combined.append(buf)
    # Apply overlap
    if overlap > 0 and len(combined) > 1:
        with_overlap = []
        prev_tail = ''
        for idx, chunk in enumerate(combined):
            if idx == 0:
                with_overlap.append(chunk)
            else:
                # take tail of previous chunk
                tail = prev_tail[-overlap:] if overlap < len(prev_tail) else prev_tail
                with_overlap.append((tail + chunk).strip())
            prev_tail = chunk
        combined = with_overlap
    return combined


def optimize_text_file(system: DualAgentAcademicSystem, file_path: str, requirements: List[str], rounds: int,
                       chunk_size: int, overlap: int, max_chunks: int = 0) -> Tuple[str, Dict]:
    """Optimize a long text file by chunking and running multi-round collaborate per chunk.
    Returns (final_combined_text, aggregated_report_dict)."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f'文本文件不存在: {file_path}')
    raw = p.read_text(encoding='utf-8')
    chunks = _split_long_text(raw, chunk_size, overlap)
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
    segment_logs = []
    optimized_segments = []
    for idx, chunk in enumerate(chunks):
        print(f'🧩 处理分段 {idx + 1}/{len(chunks)} (长度={len(chunk)})')
        final_seg, log = system.collaborate(chunk, requirements, rounds=rounds)
        optimized_segments.append(final_seg)
        segment_logs.append({
            'segment_index': idx,
            'original_length': len(chunk),
            'optimized_length': len(final_seg),
            'final_segment_text': final_seg,
            'round_logs': log
        })
    combined_final = '\n\n'.join(optimized_segments)
    aggregated = {
        'file': file_path,
        'chunks': len(chunks),
        'chunk_size': chunk_size,
        'overlap': overlap,
        'requirements': requirements,
        'final_text': combined_final,
        'segments': segment_logs,
    }
    return combined_final, aggregated


def build_arg_parser():
    p = argparse.ArgumentParser(description='Dual-agent academic optimizer (adversarial enhanced)')
    p.add_argument('command', nargs='?', default='demo', choices=['demo', 'synthesize', 'eval', 'distill'],
                   help='运行模式')
    p.add_argument('--rounds', type=int, default=2, help='协作轮次')
    p.add_argument('--text', type=str, help='自定义初始文本 (demo)')
    p.add_argument('--text-file', type=str, help='从文件读取初始文本 (长文本优化)')
    p.add_argument('--chunk-size', type=int, default=5000, help='长文本分段字符数 (默认5000, <=0 不分段)')
    p.add_argument('--chunk-overlap', type=int, default=200, help='分段重叠字符数 (默认200)')
    p.add_argument('--max-chunks', type=int, default=0, help='限制最多处理的段数 (0=不限制)')
    p.add_argument('--requirements', type=str, help='逗号/分号分隔需求列表')
    p.add_argument('--seeds-file', type=str, help='种子文本文件路径 (synthesize)')
    p.add_argument('--out', type=str, help='输出文件路径')
    p.add_argument('--lang', type=str, choices=['zh', 'en'], default='zh', help='语言')
    p.add_argument('--no-tools', action='store_true', help='禁用工具调用')
    p.add_argument('--no-memory', action='store_true', help='禁用向量记忆')
    p.add_argument('--report', type=str, help='JSON 报告输出路径')
    p.add_argument('--html-report', type=str, help='HTML 报告输出路径')
    p.add_argument('--distill-src', type=str, help='蒸馏源 JSONL (distill)')
    p.add_argument('--distill-out', type=str, help='蒸馏输出 JSONL')
    p.add_argument('--out-text-file', type=str, help='将最终优化文本写入该路径 (例如 optimized_paper.txt)')
    # HYBRID / Student model parameters
    p.add_argument('--hybrid', action='store_true', help='启用混合模式: Agent A 使用本地学生模型, Agent B 使用远程教师模型')
    p.add_argument('--student-base-model', type=str, help='学生模型 HF 名称 (覆盖 STUDENT_BASE_MODEL)')
    p.add_argument('--student-lora-dir', type=str, help='LoRA 适配器目录 (覆盖 STUDENT_LORA_DIR)')
    p.add_argument('--student-max-new-tokens', type=int, help='学生模型生成最大 new tokens (覆盖 STUDENT_MAX_NEW_TOKENS)')
    p.add_argument('--student-device', type=str, help='学生模型加载设备 (cpu/cuda 自动推断可省略)')
    p.add_argument('--student-dtype', type=str, help='学生模型 torch dtype, 例如 float16/bfloat16')
    p.add_argument('--student-device-map', type=str, help='HF device_map (auto/balanced 等)')
    p.add_argument('--student-save-config', type=str, help='保存当前学生模型配置到该 JSON 文件 (例如 data/student_last_loaded.json)')
    return p


if __name__ == '__main__':
    print('📦 环境检查: OPENAI_API_KEY={} SERPAPI_API_KEY={}'.format(bool(OPENAI_API_KEY), bool(SERPAPI_API_KEY)))
    parser = build_arg_parser()
    args = parser.parse_args()
    rounds = max(1, args.rounds)

    # 初始化系统: 根据 --hybrid 决定是否启用学生模型
    if args.hybrid:
        dual_agent_system = build_hybrid_dual_agent_system(
            base_model=args.student_base_model,
            lora_dir=args.student_lora_dir,
            max_new_tokens=args.student_max_new_tokens,
            device=args.student_device,
            torch_dtype=args.student_dtype,
            device_map=args.student_device_map,
        )
        # 保存配置（可选）
        if args.student_save_config:
            cfg_path = Path(args.student_save_config)
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            config_obj = {
                'base_model': args.student_base_model or os.getenv('STUDENT_BASE_MODEL', 'Qwen/Qwen1.5-1.8B-Chat'),
                'lora_dir': args.student_lora_dir or os.getenv('STUDENT_LORA_DIR') or '',
                'max_new_tokens': args.student_max_new_tokens or int(os.getenv('STUDENT_MAX_NEW_TOKENS', '512')),
                'device': args.student_device,
                'torch_dtype': args.student_dtype,
                'device_map': args.student_device_map,
                'saved_at': datetime.now().isoformat(),
            }
            try:
                cfg_path.write_text(json.dumps(config_obj, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'💾 已保存学生模型配置: {cfg_path}')
            except Exception as e:
                print(f'⚠️ 保存学生模型配置失败: {e}')
    else:
        dual_agent_system = DualAgentAcademicSystem(
            llm,
            TOOLS,
            vectorstore,
            enable_tools=not args.no_tools,
            enable_memory=not args.no_memory,
        )
        print('🔁 单模型模式: Agent A / B 均使用同一远程或回退 LLM')

    def _maybe_write_report(data: Dict, path: Optional[str]):
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f'📝 JSON 报告已写入: {path}')
            except Exception as e:
                print(f'⚠️ JSON 报告写入失败: {e}')


    def _maybe_write_html(final_text: str, log: List[Dict], path: Optional[str], summary: Optional[Dict] = None,
                          title: str = '多轮优化报告'):
        if path:
            try:
                html = generate_html_report(title, final_text, log, summary=summary)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(html)
                print(f'📄 HTML 报告已写入: {path}')
            except Exception as e:
                print(f'⚠️ HTML 报告写入失败: {e}')


    # 新增：将最终优化文本写入同类型文本文件的辅助函数
    def _maybe_write_text(final_text: str, path: Optional[str]):
        if path:
            try:
                out_path = Path(path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(final_text, encoding='utf-8')
                print(f'📄 文本输出已写入: {out_path}')
            except Exception as e:
                print(f'⚠️ 文本输出写入失败: {e}')


    if ENABLE_INTERACTIVE:
        print('🚀 交互模式开启')
        if args.text_file:
            # Long file path provided
            reqs = parse_requirements(args.requirements, ['学术表达提升'])
            final_text, aggregated = optimize_text_file(dual_agent_system, args.text_file, reqs, rounds=rounds,
                                                        chunk_size=args.chunk_size, overlap=args.chunk_overlap,
                                                        max_chunks=args.max_chunks)
            _maybe_write_report({'final': final_text, 'aggregated': aggregated}, args.report)
            _maybe_write_html(final_text, aggregated.get('segments', []), args.html_report, title='交互模式长文本报告')
            _maybe_write_text(final_text, args.out_text_file)
        else:
            final_text, log = dual_agent_system.collaborate(args.text or '交互模式初稿',
                                                            parse_requirements(args.requirements, ['学术表达提升']),
                                                            rounds=rounds)
            _maybe_write_report({'final': final_text, 'log': log}, args.report)
            _maybe_write_html(final_text, log, args.html_report, title='交互模式报告')
            _maybe_write_text(final_text, args.out_text_file)
    else:
        if args.command == 'demo':
            print('🚀 Demo 演示模式')
            if args.text_file:
                reqs = parse_requirements(args.requirements,
                                          ['学术表达提升', '逻辑结构优化'] if args.lang == 'zh' else ['academic polish',
                                                                                                      'logical coherence'])
                final_text, aggregated = optimize_text_file(dual_agent_system, args.text_file, reqs, rounds=rounds,
                                                            chunk_size=args.chunk_size, overlap=args.chunk_overlap,
                                                            max_chunks=args.max_chunks)
                print('\n📌 Long file optimized final text (truncated preview):\n',
                      final_text[:800] + ('...' if len(final_text) > 800 else ''))
                _maybe_write_report({'final': final_text, 'aggregated': aggregated}, args.report)
                _maybe_write_html(final_text, aggregated.get('segments', []), args.html_report, title='长文本优化报告')
                _maybe_write_text(final_text, args.out_text_file)
            else:
                base_default = (
                    'This is a preliminary draft about multi-agent collaboration in academic writing.' if args.lang == 'en' else '这是一段关于多智能体协作进行学术写作优化的初稿。'
                )
                sample_text = args.text or base_default
                reqs = parse_requirements(args.requirements,
                                          ['学术表达提升', '逻辑结构优化'] if args.lang == 'zh' else ['academic polish',
                                                                                                      'logical coherence'])
                final_text, log = dual_agent_system.collaborate(sample_text, reqs, rounds=rounds)
                print('\n📌 Final optimized text:\n', final_text)
                _maybe_write_report({'final': final_text, 'log': log}, args.report)
                _maybe_write_html(final_text, log, args.html_report, title='Demo 优化报告')
                _maybe_write_text(final_text, args.out_text_file)
        elif args.command == 'synthesize':
            print('🧪 数据合成模式')
            seeds = load_seeds_from_file(args.seeds_file) or [
                '本研究探讨了基于多智能体的文本优化框架，初步实验尚不充分。',
                '我们提出一个简单的管线，但方法部分缺乏清晰的因果论证。',
                '实验结果显示一定改进，但统计显著性需要进一步说明。',
            ]
            reqs = parse_requirements(args.requirements, ['学术表达提升', '结构清晰', '可读性增强'])
            out_path = Path(args.out) if args.out else None
            path = dual_agent_system.synthesize_dataset(seeds, reqs, rounds=rounds, out_path=out_path)
            _maybe_write_report({'dataset_path': str(path)}, args.report)
            if args.html_report:
                print('ℹ️ synthesize 模式不生成单一流程 HTML 报告，忽略 --html-report')
        elif args.command == 'eval':
            print('🧮 评估模式')
            tests = [
                ('本文提出一种方法，但存在一些问题，需要更严谨的叙述。',
                 parse_requirements(args.requirements, ['严谨性', '逻辑连贯'])),
                ('我们的实验结果较为有限，缺少消融实验。',
                 parse_requirements(args.requirements, ['补充实验建议', '学术化表达']))
            ]
            report = dual_agent_system.evaluate(tests, rounds=rounds)
            _maybe_write_report(report, args.report)
            _maybe_write_html('N/A (Eval 多案例)', report.get('cases', []), args.html_report,
                              summary=report.get('summary'), title='评估指标汇总报告')
        elif args.command == 'distill':
            print('🧪 蒸馏数据生成模式')
            src = Path(args.distill_src) if args.distill_src else Path(args.out or 'data/latest_synth.jsonl')
            if not src.exists():
                print(f'⚠️ 蒸馏源不存在: {src}')
            else:
                distill_out = Path(args.distill_out) if args.distill_out else Path('data/distill_pairs.jsonl')
                distill_out.parent.mkdir(parents=True, exist_ok=True)
                dual_agent_system.prepare_distillation_pairs(src, distill_out)
                _maybe_write_report({'distill_pairs': str(distill_out)}, args.report)
                if args.html_report:
                    print('ℹ️ distill 模式不生成 HTML 报告，忽略 --html-report')
