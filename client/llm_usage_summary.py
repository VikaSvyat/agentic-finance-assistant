import argparse
import re
from collections import defaultdict
from pathlib import Path


USAGE_BLOCK_RE = re.compile(
    r"LLM usage\s+"
    r"Model:\s*(?P<model>.+?)\s+"
    r"Prompt tokens:\s*(?P<prompt>\d+|n/a)\s+"
    r"Completion tokens:\s*(?P<completion>\d+|n/a)\s+"
    r"Total tokens:\s*(?P<total>\d+|n/a)",
    re.MULTILINE,
)


def token_value(value: str) -> int:
    return 0 if value == "n/a" else int(value)


def summarize_usage(log_path: Path) -> dict[str, dict[str, int]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )

    for match in USAGE_BLOCK_RE.finditer(text):
        model = match.group("model").strip()
        item = summary[model]
        item["calls"] += 1
        item["prompt_tokens"] += token_value(match.group("prompt"))
        item["completion_tokens"] += token_value(match.group("completion"))
        item["total_tokens"] += token_value(match.group("total"))

    return dict(summary)


def print_summary(summary: dict[str, dict[str, int]]) -> None:
    if not summary:
        print("No LLM usage entries found.")
        return

    grand_total = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for model in sorted(summary):
        item = summary[model]
        for key in grand_total:
            grand_total[key] += item[key]

        print(f"Model: {model}")
        print(f"Calls: {item['calls']:,}")
        print(f"Prompt tokens: {item['prompt_tokens']:,}")
        print(f"Completion tokens: {item['completion_tokens']:,}")
        print(f"Total tokens: {item['total_tokens']:,}")
        print()

    print("Overall total")
    print(f"Calls: {grand_total['calls']:,}")
    print(f"Prompt tokens: {grand_total['prompt_tokens']:,}")
    print(f"Completion tokens: {grand_total['completion_tokens']:,}")
    print(f"Total tokens: {grand_total['total_tokens']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LLM token usage from agent logs.")
    parser.add_argument(
        "log_path",
        nargs="?",
        default="logs/agent.log",
        help="Path to the agent log file. Defaults to logs/agent.log.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    print_summary(summarize_usage(log_path))


if __name__ == "__main__":
    main()
