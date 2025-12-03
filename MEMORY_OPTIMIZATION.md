# 内存优化指南

## 问题描述

本项目使用的本地模型虽然只有 3GB 大小，但在运行时可能会消耗 10GB 以上的内存，导致程序崩溃或自动终止。

## 问题原因分析

### 1. 模型精度问题
- **原因**：3GB 的模型文件是压缩后的权重，加载时默认使用 FP16/BF16 精度
- **内存占用**：FP16 模式下，3GB 模型实际占用约 6-8GB 内存
- **解决方案**：启用 4bit 量化，内存占用降至 1-2GB

### 2. 重复加载问题
- **原因**：Web 应用启动时立即加载模型，而非在需要时才加载
- **内存占用**：应用启动时就占用大量内存
- **解决方案**：实现延迟加载机制，仅在首次请求时加载模型

### 3. 设备映射问题
- **原因**：默认使用 GPU 模式，同时占用 GPU 显存和系统内存
- **内存占用**：GPU 显存 + CPU 内存双重占用
- **解决方案**：支持强制 CPU 模式，或优化 device_map

### 4. 内存碎片化
- **原因**：PyTorch 缓存、梯度累积等导致内存碎片
- **内存占用**：实际使用远超模型本身大小
- **解决方案**：限制最大内存使用，定期清理缓存

## 优化措施

### 已实施的优化

#### 1. 启用 4bit 量化（默认）
```python
# hf_student_llm.py
# 默认启用 4bit 量化，从环境变量读取
STUDENT_LOAD_IN_4BIT=1  # 启用（内存 ~1-2GB）
```

#### 2. 延迟加载机制
```python
# multi_agent_nlp_project.py
# 模型不在导入时加载，而是在首次使用时加载
dual_agent_system = _LazyDualAgentSystemProxy()
```

#### 3. 单例模式
```python
# app.py
# 确保模型只加载一次，避免重复实例化
def get_web_dual_agent_system():
    global _web_dual_agent_system
    if _web_dual_agent_system is None:
        # 首次加载
    return _web_dual_agent_system
```

#### 4. 内存限制
```python
# hf_student_llm.py
# 限制 GPU 和 CPU 最大内存使用
"max_memory": {0: "3GB", "cpu": "8GB"}
```

#### 5. CPU Fallback 支持
```python
# 通过环境变量强制使用 CPU 模式
STUDENT_FORCE_CPU=1
```

## 配置方案

### 方案 1: 低内存方案（推荐用于 Web 应用）
**适用场景**：内存 < 8GB 或频繁崩溃

```bash
# .env 配置
STUDENT_LOAD_IN_4BIT=1
STUDENT_FORCE_CPU=1
ENABLE_HYBRID=1
```

**预期内存占用**：~2-3GB
**性能**：较慢（CPU 推理）

### 方案 2: 标准方案
**适用场景**：内存 8-16GB，有 GPU

```bash
# .env 配置
STUDENT_LOAD_IN_4BIT=1
STUDENT_FORCE_CPU=0
ENABLE_HYBRID=1
```

**预期内存占用**：~3-5GB（GPU 显存 + 系统内存）
**性能**：良好

### 方案 3: 高性能方案
**适用场景**：内存 > 16GB，有 GPU

```bash
# .env 配置
STUDENT_LOAD_IN_4BIT=0
STUDENT_FORCE_CPU=0
ENABLE_HYBRID=1
```

**预期内存占用**：~6-10GB
**性能**：最佳

### 方案 4: 仅教师模型（最省内存）
**适用场景**：不需要本地模型，只使用在线 API

```bash
# .env 配置
ENABLE_HYBRID=0
```

**预期内存占用**：< 1GB
**性能**：取决于 API 响应速度

## 环境变量说明

### 内存优化相关

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `STUDENT_LOAD_IN_4BIT` | `1` | 启用 4bit 量化（1=启用，0=禁用） |
| `STUDENT_FORCE_CPU` | `0` | 强制 CPU 模式（1=CPU，0=GPU） |
| `ENABLE_HYBRID` | `1` | 混合模式（1=启用，0=禁用） |
| `FORCE_STUDENT_STUB` | `0` | Stub 模式（1=测试模式，0=正常） |

### 模型配置相关

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `STUDENT_BASE_MODEL` | `Qwen/Qwen1.5-1.8B-Chat` | 学生模型路径 |
| `STUDENT_LORA_DIR` | `` | LoRA 适配器路径（可选） |
| `STUDENT_MAX_NEW_TOKENS` | `512` | 最大生成长度 |

## 使用步骤

### 1. 配置环境变量

复制并编辑配置文件：
```bash
cp .env.example .env
# 编辑 .env 文件，根据内存情况选择合适的方案
```

### 2. 安装依赖

确保安装了 bitsandbytes（用于 4bit 量化）：
```bash
pip install bitsandbytes
```

### 3. 启动应用

```bash
# 启动 Web 应用
python web_interface/start_web.py

# 或使用批处理文件
start_web.bat
```

### 4. 监控内存使用

在 Windows 上：
```powershell
# 查看进程内存使用
Get-Process python | Select-Object Name, PM, WS
```

## 故障排查

### 问题 1: 仍然内存不足
**解决方案**：
1. 确认 `.env` 配置已生效
2. 尝试使用 `STUDENT_FORCE_CPU=1`
3. 考虑使用 `ENABLE_HYBRID=0`（仅教师模型）

### 问题 2: 4bit 量化失败
**症状**：提示 `bitsandbytes not available`
**解决方案**：
```bash
pip install bitsandbytes
# Windows 用户可能需要额外步骤，参考：
# https://github.com/TimDettmers/bitsandbytes/issues/
```

### 问题 3: CUDA 错误
**症状**：CUDA out of memory
**解决方案**：
```bash
# 使用 CPU 模式
STUDENT_FORCE_CPU=1
```

### 问题 4: 模型加载缓慢
**原因**：首次下载模型需要时间
**解决方案**：
1. 使用本地模型路径
2. 或等待 HuggingFace 下载完成

## 性能对比

| 方案 | 内存占用 | 加载时间 | 推理速度 | 适用场景 |
|------|----------|----------|----------|----------|
| FP16 + GPU | ~6-10GB | ~30s | 快 | 高性能服务器 |
| 4bit + GPU | ~3-5GB | ~20s | 中等 | 标准工作站 |
| 4bit + CPU | ~2-3GB | ~15s | 慢 | 低内存环境 |
| 仅教师模型 | < 1GB | < 5s | 取决于API | 云端部署 |

## 附加建议

1. **定期重启**：长时间运行后重启应用以清理内存
2. **限制并发**：Web 应用限制同时处理的请求数
3. **使用更小的模型**：考虑 0.5B 或 1B 参数的模型
4. **批量处理**：避免频繁的小批量请求

## 技术细节

### 4bit 量化原理
- 将 FP16 权重量化为 4bit 整数
- 使用 bitsandbytes 库的 NF4 量化
- 几乎不损失精度，内存减少 75%

### 延迟加载实现
```python
class _LazyDualAgentSystemProxy:
    def __getattr__(self, name):
        # 首次访问时才加载真实模型
        real_system = get_dual_agent_system()
        return getattr(real_system, name)
```

### 内存限制实现
```python
load_kwargs.update({
    "load_in_4bit": True,
    "device_map": "auto",
    "max_memory": {0: "3GB", "cpu": "8GB"},
})
```

## 参考资源

- [Hugging Face - 4bit 量化文档](https://huggingface.co/docs/transformers/main_classes/quantization)
- [bitsandbytes GitHub](https://github.com/TimDettmers/bitsandbytes)
- [PyTorch 内存管理](https://pytorch.org/docs/stable/notes/cuda.html)
