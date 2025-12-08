import argparse
import json
import re
from pathlib import Path


def clean_input_text(raw: str) -> str:
    """Extract the real user paragraph from a LangChain-style content blob."""
    if not isinstance(raw, str):
        return raw  # unexpected type; leave as-is

    # Prefer explicit content='...'
    m = re.search(r"content='([^']*)'", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r'content="([^"]*)"', raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    # Fallback: drop everything after additional_kwargs/response_metadata if present
    for marker in ("additional_kwargs", "response_metadata", "usage_metadata"):
        if marker in raw:
            return raw.split(marker, 1)[0].replace("content=", "").strip().strip("'\"")

    # If nothing matched, return original
    return raw.strip()


def process_file(in_path: Path, out_path: Path) -> tuple[int, int]:
    cleaned = 0
    total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ 跳过无法解析的行: {line[:120]}...")
                continue
            old = obj.get("input", "")
            new = clean_input_text(old)
            if new != old:
                cleaned += 1
                obj["input"] = new
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return cleaned, total


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗 teacher_only JSONL 中被 LLM 元数据包裹的 input 字段")
    parser.add_argument("--input", default="data/teacher_only_3k.jsonl", help="原始 JSONL 路径")
    parser.add_argument("--output", default="data/teacher_only_3k.cleaned.jsonl", help="清洗后输出路径")
    parser.add_argument("--in-place", action="store_true", help="直接覆盖输入文件")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"输入文件不存在: {in_path}")

    if args.in_place:
        tmp_out = in_path.with_suffix(in_path.suffix + ".tmp")
        cleaned, total = process_file(in_path, tmp_out)
        tmp_out.replace(in_path)
        print(f"✅ 已就地清洗: {in_path} | 共 {total} 行，修改 {cleaned} 行")
    else:
        cleaned, total = process_file(in_path, out_path)
        print(f"✅ 已写出清洗文件: {out_path} | 共 {total} 行，修改 {cleaned} 行")


if __name__ == "__main__":
    main()
