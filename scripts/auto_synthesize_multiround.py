import argparse
import math
import random
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multi_agent_nlp_project import DualAgentAcademicSystem, llm, parse_requirements


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

_SEED_TEMPLATES = [
    "在{domain}领域的实践中，当前方案因{pain}而表现不足，导致{impact}。请写一段学术化的背景描述，概括研究动机与现实困境。",
    "本研究聚焦{domain}场景下的关键问题，即{pain}。试撰写一段研究引言，说明问题的重要性、现有不足以及拟解决方向。",
    "面向{domain}应用，本项目试图应对{pain}带来的挑战，请用学术语体写一段段落，交代研究背景、实践痛点与潜在贡献。",
]

_PAIN_POINTS = [
    "缺乏结构化知识图谱支撑",
    "数据标注与清洗成本过高",
    "模型决策过程可解释性不足",
    "多源异构数据难以有效融合",
    "跨场景迁移与泛化能力有限",
    "长期监测与持续评估机制缺失",
    "决策流程中多主体协同效率低",
]

_IMPACTS = [
    "系统性能与可靠性难以满足实际需求",
    "难以及时支撑关键业务决策",
    "用户体验割裂，信任度下降",
    "资源配置效率低下，运营成本上升",
    "难以形成可复用的方法论与技术路线",
]


def _parse_domains(raw: str | None) -> List[str]:
    if not raw:
        return _DEFAULT_DOMAINS
    parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
    return parts or _DEFAULT_DOMAINS


def _rule_seeds(count: int, domain_str: str | None) -> List[str]:
    domains = _parse_domains(domain_str)
    seeds: List[str] = []
    for _ in range(count):
        d = random.choice(domains)
        pain = random.choice(_PAIN_POINTS)
        impact = random.choice(_IMPACTS)
        tpl = random.choice(_SEED_TEMPLATES)
        seeds.append(tpl.format(domain=d, pain=pain, impact=impact))
    return seeds


def _llm_seeds(count: int, domain_str: str | None, requirements: List[str]) -> List[str]:
    if count <= 0:
        return []
    domains = _parse_domains(domain_str)
    seeds: List[str] = []
    sys_prompt = (
        "你是学术写作助手，请为不同应用领域生成需要优化的中文学术段落。"
        "每段应包含：研究背景、现实痛点、方法或技术路径、预期贡献。"
        "不写小标题，无需分点，直接输出一段 80-200 字的学术语体文字。"
    )
    focus_req = requirements[0] if requirements else "学术表达提升"
    for i in range(count):
        d = random.choice(domains)
        user = (
            f"领域: {d}\n"
            f"写作要求: {focus_req}\n"
            f"请生成第 {i+1} 个需要进行学术化改写的草稿段落。"
        )
        composed = f"系统指令: {sys_prompt}\n\n用户指令: {user}"
        resp = llm.invoke(composed)
        text = resp.strip() if isinstance(resp, str) else str(resp)
        seeds.append(text)
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="自动生成多轮 synth 数据：规则模板 + LLM seeds，不依赖 seeds.txt")
    parser.add_argument("--count", type=int, default=100, help="目标样本数量（近似）")
    parser.add_argument("--rounds", type=int, default=2, help="多轮协作轮数")
    parser.add_argument("--requirements", type=str,
                        default="学术表达提升;结构清晰;可读性增强",
                        help="要求列表，逗号/分号分隔")
    parser.add_argument("--rule-ratio", type=float, default=0.5,
                        help="规则模板 seeds 占比，0-1 之间")
    parser.add_argument("--domains", type=str,
                        help="逗号分隔的领域列表，用于构造 seeds")
    parser.add_argument("--out", type=str,
                        default="data/synth_auto_multiround.jsonl",
                        help="输出 JSONL 路径")

    args = parser.parse_args()

    total = max(1, args.count)
    rule_n = max(0, min(total, int(total * args.rule_ratio)))
    llm_n = max(0, total - rule_n)

    reqs = parse_requirements(args.requirements, ["学术表达提升", "结构清晰", "可读性增强"])

    print(f"📦 计划生成约 {total} 个种子：规则模板 {rule_n} 个，LLM {llm_n} 个…")
    rule_seeds = _rule_seeds(rule_n, args.domains)
    llm_based = _llm_seeds(llm_n, args.domains, reqs)
    seeds = rule_seeds + llm_based
    random.shuffle(seeds)

    # 若 seeds 少于 count，则重复使用，保持近似数量
    if len(seeds) < total:
        repeats = math.ceil(total / max(1, len(seeds)))
        seeds = (seeds * repeats)[:total]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    system = DualAgentAcademicSystem(llm, [], None)
    print(f"🚀 启动多轮协作生成，轮数={args.rounds}，输出={out_path}")
    system.synthesize_dataset(seeds=seeds, requirements=reqs, rounds=args.rounds, out_path=out_path)


if __name__ == "__main__":
    main()
