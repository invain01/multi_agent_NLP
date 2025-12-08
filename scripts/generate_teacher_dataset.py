import argparse
import math
import random
import re
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multi_agent_nlp_project import (
    generate_teacher_only_dataset,
    load_seeds_from_file,
    parse_requirements,
    llm,
)


def _expand_seeds(seeds: List[str], samples_per_seed: int) -> List[str]:
    if samples_per_seed <= 1:
        return seeds
    expanded: List[str] = []
    for seed in seeds:
        for idx in range(samples_per_seed):
            hint = f"\n\n变体提示: 请生成与原意一致的第{idx + 1}个不同版本，突出不同细节。"
            expanded.append(seed + hint)
    return expanded


_DEFAULT_DOMAINS = [
    "智能医疗",
    "低碳交通",
    "工业质检",
    "教育评测",
    "灾害预警",
    "科研写作",
    "法律审查",
    "金融风控",
    "供应链调度",
    "文化创意",
    "智慧农业",
    "公共卫生",
    "航天测控",
    "智能制造",
    "文物修复",
    "智慧城市治理",
    "新能源运维",
    "跨境电商",
    "环境监测",
    "心理健康辅导",
    "体育竞技分析",
    "海洋探测",
    "智慧养老",
    "危化品监管",
]

_PAIN_POINTS = [
    "缺乏结构化知识图谱",
    "数据标注投入过高",
    "跨语言协同存在障碍",
    "实时决策缺少可解释性",
    "多模态信号融合精度不足",
    "制度合规审查流程冗长",
    "用户体验割裂",
]

_IMPROVEMENTS = [
    "构建可复用的领域知识蒸馏链路",
    "引入对抗式评审强化事实一致性",
    "利用小模型联邦协作降低推理成本",
    "叠加记忆检索以保持上下文连贯",
    "自动生成实验报告与可视化图表",
    "形成闭环监控指标体系",
]

_DELIVERABLES = [
    "多轮交互脚本",
    "自监督训练集",
    "对话式质检模板",
    "跨域问答基准",
    "评估指标看板",
    "轻量部署方案",
]

_SEED_TEMPLATES = [
    "本研究聚焦{domain}任务，源于当前系统{pain}，为此拟{improve}，并计划交付{deliver}，以验证学术与工程价值。请围绕第{case_id}个案例撰写背景与挑战。",
    "在{domain}场景中，我们观察到{pain}导致关键流程受阻。项目计划{improve}，并形成{deliver}支撑多智能体协作，需描述案例{case_id}的痛点与研究目标。",
    "针对{domain}领域的产业实践，现有方案因{pain}而表现欠佳。本轮方案试图{improve}，并推出{deliver}以支撑论文与产品双场景，请说明第{case_id}号课题的动机与难点。",
    "面向{domain}治理需求，团队正在处理{pain}带来的连锁影响。我们计划{improve}，同时产出{deliver}，请撰写与案例{case_id}对应的背景、挑战与验证路径。",
    "{domain}相关的研究计划中，核心矛盾集中在{pain}。项目拟通过{improve}加以缓解，并构建{deliver}作为关键产物。请就第{case_id}个试点梳理问题起点、研究思路与预期贡献。",
    "在{domain}这一跨学科领域，我们需要解决{pain}，计划采用{improve}并交付{deliver}。请以案例{case_id}为例，描述数据来源、协作方式与落地目标。"
]


def _parse_domain_overrides(raw: str | None) -> List[str]:
    if not raw:
        return _DEFAULT_DOMAINS
    parts = [p.strip() for p in re.split(r"[;,；]", raw) if p.strip()]
    return parts or _DEFAULT_DOMAINS


def _auto_generate_seeds(count: int, domain_str: str | None = None) -> List[str]:
    domains = _parse_domain_overrides(domain_str)
    seeds: List[str] = []
    for idx in range(count):
        domain = random.choice(domains)
        pain = random.choice(_PAIN_POINTS)
        improve = random.choice(_IMPROVEMENTS)
        deliver = random.choice(_DELIVERABLES)
        template = random.choice(_SEED_TEMPLATES)
        seeds.append(template.format(domain=domain, pain=pain, improve=improve,
                                      deliver=deliver, case_id=idx + 1))
    return seeds


def _llm_generate_seeds(count: int, domain_str: str | None, requirements: List[str]) -> List[str]:
    domains = _parse_domain_overrides(domain_str)
    seeds: List[str] = []
    sys_prompt = (
        "你是多领域写作规划助手，请随机构造学术项目背景段落，使其适合交给教师模型进行润色。"
        "每段需包含: 背景/动机、痛点、拟采取的方法或工具、预期产出。保持中文学术语体，不要引用真实机构。"
    )
    for idx in range(count):
        domain = random.choice(domains)
        focus_req = random.choice(requirements) if requirements else "学术表达提升"
        prompt = (
            f"领域: {domain}\n"
            f"要求: {focus_req}\n"
            f"请写一个独立段落，用于案例{idx + 1}，不要使用固定模板，不少于80字。"
        )
        composed = f"系统指令: {sys_prompt}\n\n用户指令: {prompt}"
        resp = llm.invoke(composed)
        text = resp.strip() if isinstance(resp, str) else str(resp)
        seeds.append(text)
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量调用 Agent B 生成教师示范数据集")
    parser.add_argument("--seeds-file", type=str, help="包含初始段落的文本文件，一行一条")
    parser.add_argument("--text", type=str, help="当未提供 seeds-file 时使用的单段文本")
    parser.add_argument("--requirements", type=str,
                        default="学术表达提升;结构清晰;可读性增强",
                        help="要求列表，使用逗号/分号分隔")
    parser.add_argument("--samples-per-seed", type=int, default=1,
                        help="每个种子生成的教师示范数量")
    parser.add_argument("--target-count", type=int, default=0,
                        help="希望生成的总样本数，0 表示不限制")
    parser.add_argument("--auto-seed-count", type=int, default=0,
                        help="自动生成的种子数量，配合 --auto-seed-domains 可选")
    parser.add_argument("--auto-seed-domains", type=str,
                        help="自动种子的领域列表，逗号/分号分隔；若缺省使用内置模板")
    parser.add_argument("--llm-seed-count", type=int, default=0,
                        help="让大模型随机生成的种子数量")
    parser.add_argument("--llm-seed-domains", type=str,
                        help="大模型随机种子的领域列表，逗号/分号分隔")
    parser.add_argument("--out", type=str, default="data/teacher_only_dataset.jsonl",
                        help="输出 JSONL 路径")
    parser.add_argument("--teacher-cache", type=str,
                        default="data/teacher_cache.jsonl",
                        help="教师示范缓存文件，避免重复调用 API")
    parser.add_argument("--append", action="store_true",
                        help="若指定，则在已有 JSONL 后追加，而非覆盖")
    args = parser.parse_args()

    seeds = load_seeds_from_file(args.seeds_file) if args.seeds_file else []
    if args.auto_seed_count:
        auto = _auto_generate_seeds(args.auto_seed_count, args.auto_seed_domains)
        seeds.extend(auto)
    if args.llm_seed_count:
        reqs_for_seed = parse_requirements(args.requirements, [])
        llm_seeds = _llm_generate_seeds(args.llm_seed_count, args.llm_seed_domains, reqs_for_seed)
        seeds.extend(llm_seeds)

    if not seeds and args.text:
        seeds = [args.text]
    if not seeds:
        raise SystemExit("缺少 seeds 或 text，无法生成数据集")

    if args.target_count and args.target_count > len(seeds) * args.samples_per_seed:
        repeats = math.ceil(args.target_count / (len(seeds) * args.samples_per_seed))
        seeds = seeds * repeats

    expanded = _expand_seeds(seeds, args.samples_per_seed)
    if args.target_count:
        expanded = expanded[:args.target_count]

    reqs = parse_requirements(args.requirements, [])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.teacher_cache) if args.teacher_cache else None

    start_id = 0
    if args.append and out_path.exists():
        with out_path.open('r', encoding='utf-8') as existing_file:
            start_id = sum(1 for line in existing_file if line.strip())

    generate_teacher_only_dataset(
        teacher_llm=llm,
        seeds=expanded,
        requirements=reqs,
        out_path=out_path,
        cache_path=cache_path,
        start_index=start_id,
        append=args.append,
    )


if __name__ == "__main__":
    main()
