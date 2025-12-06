import os
from typing import Dict, Union, Optional

# 尝试导入 torch 与 transformers，失败则标记不可用并使用占位实现
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    # 1. 【修改点】在这里增加了 BitsAndBytesConfig 的导入
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    BitsAndBytesConfig = None # type: ignore
    _TRANSFORMERS_AVAILABLE = False

try:
    from peft import PeftModel  # type: ignore
    _PEFT_AVAILABLE = True
except Exception:
    PeftModel = None  # type: ignore
    _PEFT_AVAILABLE = False

# 尝试导入 bitsandbytes 以支持 4bit 量化推理
try:
    import bitsandbytes as bnb  # type: ignore  # noqa: F401
    _BNB_AVAILABLE = True
except Exception:
    _BNB_AVAILABLE = False

_FORCE_STUB = os.getenv("FORCE_STUDENT_STUB") == "1"
_NEED_STUB = _FORCE_STUB or (not _TORCH_AVAILABLE) or (not _TRANSFORMERS_AVAILABLE)

if _NEED_STUB:
    # ========== 轻量 Stub 模式 ==========
    # 在 CI / 无网络 / 单元测试场景下，或缺失核心依赖时，使用占位模型避免导入错误
    class HFChatLLM:  # type: ignore
        def __init__(self, base_model: str, lora_dir: Optional[str] = None, max_new_tokens: int = 512, **_):
            self.base_model = base_model
            self.lora_dir = lora_dir or ""
            self.max_new_tokens = max_new_tokens
            self.model_name = "student-stub"
        def _format_prompt(self, obj: Union[Dict, str]) -> str:
            if isinstance(obj, dict):
                return "\n".join(f"{k}: {v}" for k, v in obj.items())
            return str(obj)
        def invoke(self, prompt: Union[Dict, str]) -> str:
            text = self._format_prompt(prompt)
            # 简化输出，仅截断 + 标记
            head = text[: min(200, len(text))]
            return f"[STUB_GENERATION]\n{text[:120]}..."
        def __call__(self, prompt: Union[Dict, str]):  # 使其可调用
            return self.invoke(prompt)
        def __or__(self, other):  # 简单链式兼容：直接返回下游对象
            return other
else:
    class HFChatLLM:
        """HF + 可选 LoRA 聊天式学生模型封装 (.invoke)
        支持本地 Qwen/Qwen1.5-1.8B-Chat 及 LoRA 适配器加载。
        依赖: torch + transformers (+ peft 可选, + bitsandbytes 可选用于 4bit 推理)
        """

        def __init__(
            self,
            base_model: str,
            lora_dir: Optional[str] = None,
            max_new_tokens: int = 512,
            device: Optional[str] = None,
            torch_dtype: Optional[str] = None,
            device_map: Optional[str] = None,
            load_in_4bit: bool = None,
        ) -> None:
            self.base_model = base_model
            self.lora_dir = lora_dir or ""
            # 为了控制显存占用，限制单次生成长度
            self.max_new_tokens = min(max_new_tokens, 512)
            self.model_name = base_model

            # 是否启用 4bit 量化（默认启用以节省内存）
            if load_in_4bit is None:
                # 默认值从环境变量读取，优先启用4bit量化
                load_in_4bit = os.getenv("STUDENT_LOAD_IN_4BIT", "1") == "1"
            self.load_in_4bit = bool(load_in_4bit and _BNB_AVAILABLE and torch and torch.cuda.is_available())
            if load_in_4bit and not self.load_in_4bit:
                print("⚠️ Requested load_in_4bit=True but bitsandbytes / CUDA 不可用，回退到半精度模式。")

            if device is None:
                # 支持通过环境变量强制使用CPU模式以节省显存
                force_cpu = os.getenv("STUDENT_FORCE_CPU", "0") == "1"
                if force_cpu:
                    device = "cpu"
                    print("ℹ️ STUDENT_FORCE_CPU=1: 使用CPU模式运行学生模型（内存优化）")
                else:
                    device = "cuda" if (torch and getattr(torch, 'cuda', None) and torch.cuda.is_available()) else "cpu"
            self.device = torch.device(device) if torch else device

            # dtype 选择：默认在 GPU 上优先用 bfloat16/float16 以节省显存
            dtype = None
            if torch and torch_dtype:
                try:
                    dtype = getattr(torch, torch_dtype)
                except Exception:
                    dtype = None
            elif torch and device == "cuda": # 稍微调整了逻辑，让 dtype 也能用于 4bit config
                # 如果用户没显式指定 dtype，则在 CUDA 上尽量用半精度
                if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16

            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            load_kwargs = {"trust_remote_code": True}

            # 2. 【修改点】核心逻辑修改：使用 BitsAndBytesConfig 替代 load_in_4bit=True
            if self.load_in_4bit:
                # 创建配置对象
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype if dtype else torch.float16
                )
                
                load_kwargs.update({
                    "quantization_config": bnb_config, # 替换掉了原来的 "load_in_4bit": True
                    "device_map": device_map or "auto",
                    "max_memory": {0: "3GB", "cpu": "8GB"},  # 限制GPU和CPU内存使用
                })
                print(f"✅ 启用4bit量化加载 (BitsAndBytesConfig)，预计内存占用: ~1-2GB")
                # 4bit 模式下由 HF 管理设备分配，不再手动 model.to(device)
            else:
                if dtype is not None:
                    load_kwargs["torch_dtype"] = dtype
                # 非4bit模式下，仅在明确指定时使用device_map，避免不必要的内存映射
                if device_map:
                    load_kwargs["device_map"] = device_map

            print(f"🔄 Loading base model: {base_model}...")
            base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

            if self.lora_dir and os.path.exists(self.lora_dir) and _PEFT_AVAILABLE:
                try:
                    print(f"🔄 Loading LoRA adapter: {self.lora_dir}...")
                    self.model = PeftModel.from_pretrained(base, self.lora_dir)
                    print(f"✅ LoRA adapters loaded successfully.")
                except Exception as e:
                    print(f"⚠️ Failed to load LoRA adapters from {self.lora_dir}: {e}. Using base model.")
                    self.model = base
            else:
                if self.lora_dir and not os.path.exists(self.lora_dir):
                    print(f"⚠️ LoRA dir not found: {self.lora_dir}. Using base model.")
                if self.lora_dir and not _PEFT_AVAILABLE:
                    print("⚠️ peft not available; cannot load LoRA. Using base model.")
                self.model = base

            if not self.load_in_4bit:
                # 仅在非 4bit 模式下手动搬运到指定设备；4bit + device_map=auto 由 HF 管理
                try:
                    self.model.to(self.device)
                except Exception:
                    pass
            self.model.eval()

        def invoke(self, prompt: Union[Dict, str]) -> str:
            """执行一次推理调用，使用 Chat 模板以确保指令遵循。"""
            
            # 1. 解析输入内容
            content = ""
            if isinstance(prompt, dict):
                content = "\n".join(f"{k}: {v}" for k, v in prompt.items())
            else:
                content = str(prompt)

            # 2. 构建符合 Chat 模型的对话历史
            # 增加 System Prompt 强制规定输出格式，防止提取失败
            messages = [
                {"role": "system", "content": "你是一个学术写作优化助手。请严格遵守格式要求，输出 **优化版本：** 和 **修改说明：**。"},
                {"role": "user", "content": content}
            ]

            # 3. 使用 Tokenizer 的 apply_chat_template 进行格式化
            # 这会添加 <|im_start|>system...<|im_start|>user 等特殊 token
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(text, return_tensors="pt")
            
            if torch:
                try:
                    # 在 4bit + device_map 模式下，inputs 仍然需要放到主设备或相应 CUDA
                    inputs = inputs.to(self.device) if not self.load_in_4bit else inputs.to("cuda" if torch.cuda.is_available() else self.device)
                except Exception:
                    pass
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,      # 开启采样以获得更自然的改写
                        temperature=0.3,     # 低温度保证学术严谨性
                        top_p=0.9,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                # 只保留新生成的部分
                gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                return self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            else:
                return "[Torch unavailable: stubbed generation]"

        def __call__(self, prompt: Union[Dict, str]):  # 供 LangChain Runnable 检测
            return self.invoke(prompt)

        def __or__(self, other):  # 保持与 PromptTemplate | LLM | Parser 链式兼容
            return other