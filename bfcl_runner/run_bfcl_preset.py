#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "bfcl_runner" / "run_bfcl_eval.py"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BFCL preset runner. Select a benchmark by name without manually wiring dataset paths."
    )
    parser.add_argument("--base-url", required=True, help="Model API base URL, for example http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="Model name passed to /chat/completions")
    parser.add_argument("--benchmark", required=True, help="Benchmark name such as multi_turn, multi_turn_base, single_turn, live, non_live, multiple, parallel, etc.")
    parser.add_argument("--output-dir", required=True, help="Directory used to save summary, results, and traces.")
    parser.add_argument("--api-key", default="", help="Optional model API key.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--split", default="val")
    parser.add_argument("--task-ids", default="", help="Comma-separated task ids. Overrides split selection when set.")
    parser.add_argument("--task-contains", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--disable-tools-api", action="store_true")
    parser.add_argument("--is-open-query", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-save-trace", action="store_true")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--benchmark",
        args.benchmark,
        "--output-dir",
        args.output_dir,
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--split",
        args.split,
        "--max-turns",
        str(args.max_turns),
        "--parallelism",
        str(args.parallelism),
    ]

    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if args.task_ids:
        command.extend(["--task-ids", args.task_ids])
    if args.task_contains:
        command.extend(["--task-contains", args.task_contains])
    if args.max_tasks is not None:
        command.extend(["--max-tasks", str(args.max_tasks)])
    if args.disable_tools_api:
        command.append("--disable-tools-api")
    if args.is_open_query:
        command.append("--is-open-query")
    if args.stop_on_error:
        command.append("--stop-on-error")
    if args.no_save_trace:
        command.append("--no-save-trace")

    print(json.dumps({"command": command}, ensure_ascii=False, indent=2))
    completed = subprocess.run(command, cwd=REPO_ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
