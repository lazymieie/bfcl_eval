#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BFCL_ENV_DIR = REPO_ROOT / "env_service" / "environments" / "bfcl"
BFCL_DATA_DIR = BFCL_ENV_DIR / "bfcl_data"
BFCL_ANSWER_DIR = BFCL_ENV_DIR / "bfcl_eval" / "possible_answer"

_BENCHMARK_ALIASES = {
    "multiturn": "multi_turn",
    "singleturn": "single_turn",
    "nonlive": "non_live",
}

_BENCHMARK_PRESETS: Dict[str, Dict[str, Any]] = {
    "multi_turn": {
        "data_path": BFCL_DATA_DIR / "multi_turn_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "multi_turn_split_ids.json",
    },
    "multi_turn_base": {
        "data_path": BFCL_DATA_DIR / "multi_turn_base_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "multi_turn_base_split_ids.json",
    },
    "multi_turn_miss_func": {
        "data_path": BFCL_DATA_DIR / "multi_turn_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "multi_turn_split_ids.json",
        "task_prefix": "multi_turn_miss_func_",
    },
    "multi_turn_miss_param": {
        "data_path": BFCL_DATA_DIR / "multi_turn_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "multi_turn_split_ids.json",
        "task_prefix": "multi_turn_miss_param_",
    },
    "multi_turn_long_context": {
        "data_path": BFCL_DATA_DIR / "multi_turn_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "multi_turn_split_ids.json",
        "task_prefix": "multi_turn_long_context_",
    },
    "single_turn": {
        "data_path": BFCL_DATA_DIR / "single_turn_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "single_turn_split_ids.json",
    },
    "live": {
        "data_path": BFCL_DATA_DIR / "live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "live_split_ids.json",
    },
    "live_simple": {
        "data_path": BFCL_DATA_DIR / "live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "live_split_ids.json",
        "task_prefix": "live_simple_",
    },
    "live_multiple": {
        "data_path": BFCL_DATA_DIR / "live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "live_split_ids.json",
        "task_prefix": "live_multiple_",
    },
    "live_parallel": {
        "data_path": BFCL_DATA_DIR / "live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "live_split_ids.json",
        "task_prefix": "live_parallel_",
    },
    "live_parallel_multiple": {
        "data_path": BFCL_DATA_DIR / "live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "live_split_ids.json",
        "task_prefix": "live_parallel_multiple_",
    },
    "live_irrelevance": {
        "data_path": BFCL_DATA_DIR / "live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "live_split_ids.json",
        "task_prefix": "live_irrelevance_",
    },
    "live_relevance": {
        "data_path": BFCL_DATA_DIR / "live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "live_split_ids.json",
        "task_prefix": "live_relevance_",
    },
    "non_live": {
        "data_path": BFCL_DATA_DIR / "non_live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "non_live_split_ids.json",
    },
    "simple": {
        "data_path": BFCL_DATA_DIR / "non_live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "non_live_split_ids.json",
        "task_prefix": "simple_python_",
    },
    "simple_python": {
        "data_path": BFCL_DATA_DIR / "non_live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "non_live_split_ids.json",
        "task_prefix": "simple_python_",
    },
    "simple_java": {
        "data_path": BFCL_DATA_DIR / "non_live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "non_live_split_ids.json",
        "task_prefix": "simple_java_",
    },
    "simple_javascript": {
        "data_path": BFCL_DATA_DIR / "non_live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "non_live_split_ids.json",
        "task_prefix": "simple_javascript_",
    },
    "multiple": {
        "data_path": BFCL_DATA_DIR / "non_live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "non_live_split_ids.json",
        "task_prefix": "multiple_",
    },
    "parallel": {
        "data_path": BFCL_DATA_DIR / "non_live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "non_live_split_ids.json",
        "task_prefix": "parallel_",
    },
    "parallel_multiple": {
        "data_path": BFCL_DATA_DIR / "non_live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "non_live_split_ids.json",
        "task_prefix": "parallel_multiple_",
    },
    "irrelevance": {
        "data_path": BFCL_DATA_DIR / "non_live_processed.jsonl",
        "split_ids_path": BFCL_DATA_DIR / "non_live_split_ids.json",
        "task_prefix": "irrelevance_",
    },
}


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda match: os.getenv(match.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return _expand_env_vars(json.load(file))


def _write_json(path: Path, data: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_bfcl_env_cls():
    from env_service.environments.bfcl.bfcl_env import BfclEnv

    return BfclEnv


def _tool_calls_to_text(tool_calls: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = _safe_json_loads(arguments, arguments)
        payload = {
            "name": function.get("name", ""),
            "arguments": arguments,
        }
        blocks.append(f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>")
    return "\n".join(blocks)


def _coerce_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    return str(content)


def _assistant_message_for_history(message: Dict[str, Any]) -> Dict[str, str]:
    content = _coerce_message_content(message.get("content"))
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        tool_text = _tool_calls_to_text(tool_calls)
        if content:
            content = f"{content.rstrip()}\n{tool_text}"
        else:
            content = tool_text
    return {"role": "assistant", "content": content}


def _safe_json_loads(raw: Any, fallback: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return fallback
    return raw


def _normalize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        normalized.append(
            {
                "id": tool_call.get("id", f"tool_call_{index}"),
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": function.get("name", ""),
                    "arguments": arguments,
                },
            }
        )
    return normalized


def _resolve_path(base_dir: Path, raw_value: Optional[str], *, required: bool) -> Optional[Path]:
    if not raw_value:
        if required:
            raise ValueError("Missing required path value")
        return None
    path = Path(raw_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if required and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def _read_split_ids(path: Path, split: str) -> List[str]:
    data = _read_json(path)
    if split not in data:
        raise KeyError(f"Split '{split}' not found in {path}")
    task_ids = data[split]
    if not isinstance(task_ids, list):
        raise TypeError(f"Split '{split}' in {path} must be a list")
    return [str(task_id) for task_id in task_ids]


def _normalize_benchmark_name(raw_name: Optional[str]) -> Optional[str]:
    if raw_name is None:
        return None
    benchmark = str(raw_name).strip().lower()
    if not benchmark:
        return None
    return _BENCHMARK_ALIASES.get(benchmark, benchmark)


def _resolve_benchmark_preset(raw_name: Optional[str]) -> Dict[str, Any]:
    benchmark = _normalize_benchmark_name(raw_name)
    if benchmark is None:
        return {}
    if benchmark not in _BENCHMARK_PRESETS:
        supported = ", ".join(sorted(_BENCHMARK_PRESETS))
        raise ValueError(f"Unsupported BFCL benchmark '{raw_name}'. Supported values: {supported}")

    preset = dict(_BENCHMARK_PRESETS[benchmark])
    preset["benchmark"] = benchmark
    preset.setdefault("answer_path", BFCL_ANSWER_DIR)
    preset.setdefault("split", "val")
    return preset


def _select_task_ids(config: "RunnerConfig") -> List[str]:
    if config.bfcl.task_ids:
        task_ids = [str(task_id) for task_id in config.bfcl.task_ids]
    else:
        task_ids = _read_split_ids(config.bfcl.split_ids_path, config.bfcl.split)

    if config.bfcl.task_prefix:
        task_ids = [task_id for task_id in task_ids if task_id.startswith(config.bfcl.task_prefix)]

    if config.bfcl.task_contains:
        task_ids = [task_id for task_id in task_ids if config.bfcl.task_contains in task_id]

    if config.bfcl.max_tasks is not None:
        task_ids = task_ids[: config.bfcl.max_tasks]

    if not task_ids:
        selection = config.bfcl.benchmark or config.bfcl.task_prefix or config.bfcl.task_contains or config.bfcl.split
        raise ValueError(f"No BFCL tasks matched the current selection: {selection}")

    return task_ids


def _coerce_usage(payload: Dict[str, Any]) -> Dict[str, int]:
    usage = payload.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


@dataclass
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 180
    use_tools_api: bool = True
    extra_body: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class BfclConfig:
    data_path: Path
    answer_path: Path
    split_ids_path: Path
    benchmark: Optional[str] = None
    split: str = "test"
    task_ids: List[str] = field(default_factory=list)
    task_prefix: Optional[str] = None
    task_contains: Optional[str] = None
    max_tasks: Optional[int] = None
    is_open_query: bool = False


@dataclass
class RunConfig:
    output_dir: Path
    max_turns: int = 24
    stop_on_error: bool = False
    save_trace: bool = True
    parallelism: int = 1


@dataclass
class RunnerConfig:
    model: ModelConfig
    bfcl: BfclConfig
    run: RunConfig
    raw_config: Dict[str, Any]


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.url = f"{config.base_url.rstrip('/')}/chat/completions"

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if self.config.use_tools_api and tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.config.extra_body:
            payload.update(copy.deepcopy(self.config.extra_body))

        req = request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._build_headers(),
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.config.timeout) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model API returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Model API request failed: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model API returned non-JSON response: {body[:500]}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Model API response does not contain choices: {_json_dumps(data)}")

        message = choices[0].get("message") or {}
        assistant_message = {
            "role": "assistant",
            "content": _coerce_message_content(message.get("content")),
        }
        if message.get("tool_calls"):
            assistant_message["tool_calls"] = _normalize_tool_calls(message["tool_calls"])

        return {
            "assistant_message": assistant_message,
            "finish_reason": choices[0].get("finish_reason"),
            "usage": _coerce_usage(data),
            "raw_response": data,
        }

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            **self.config.headers,
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers


class BfclEvaluatorRunner:
    def __init__(self, config: RunnerConfig):
        self.config = config
        self.client = OpenAICompatibleClient(config.model)
        self.results: List[Dict[str, Any]] = []

    def _current_tools(self, env: Any) -> List[Dict[str, Any]]:
        if env.env_handler is None or env.original_test_entry is None:
            return []
        return env.env_handler._compile_tools(env.original_test_entry)

    def _create_env(self, task_id: str):
        env_cls = _load_bfcl_env_cls()
        env = env_cls(
            task_id=task_id,
            instance_id=f"bfcl_runner_{int(time.time())}_{task_id}",
            params={
                "data_path": str(self.config.bfcl.data_path),
                "answer_path": str(self.config.bfcl.answer_path),
                "model_name": self.config.model.model,
                "is_open_query": self.config.bfcl.is_open_query,
            },
        )
        return env

    def run(self) -> Dict[str, Any]:
        _ensure_dir(self.config.run.output_dir)
        task_ids = _select_task_ids(self.config)
        started_at = time.time()
        indexed_results: List[Optional[Dict[str, Any]]] = [None] * len(task_ids)

        if self.config.run.parallelism <= 1:
            for index, task_id in enumerate(task_ids, start=1):
                print(f"[{index}/{len(task_ids)}] Running {task_id} ...", flush=True)
                result = self.run_single_task(task_id)
                indexed_results[index - 1] = result
                summary_text = (
                    f"score={result.get('score', 0.0):.4f}, "
                    f"turns={result.get('turns_completed', 0)}, "
                    f"status={result.get('status')}"
                )
                print(f"  -> {summary_text}", flush=True)
                if result["status"] == "error" and self.config.run.stop_on_error:
                    raise RuntimeError(f"Stopped on error for task {task_id}")
        else:
            print(
                f"Running {len(task_ids)} BFCL tasks with {self.config.run.parallelism} threads ...",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=self.config.run.parallelism) as executor:
                future_to_meta = {
                    executor.submit(self.run_single_task, task_id): (index, task_id)
                    for index, task_id in enumerate(task_ids, start=1)
                }
                completed = 0
                for future in as_completed(future_to_meta):
                    index, task_id = future_to_meta[future]
                    result = future.result()
                    indexed_results[index - 1] = result
                    completed += 1
                    summary_text = (
                        f"score={result.get('score', 0.0):.4f}, "
                        f"turns={result.get('turns_completed', 0)}, "
                        f"status={result.get('status')}"
                    )
                    print(f"[{completed}/{len(task_ids)}] Finished {task_id} -> {summary_text}", flush=True)
                    if result["status"] == "error" and self.config.run.stop_on_error:
                        raise RuntimeError(f"Stopped on error for task {task_id}")

        self.results = [result for result in indexed_results if result is not None]

        summary = self._build_summary(task_ids, started_at)
        _write_jsonl(self.config.run.output_dir / "results.jsonl", self.results)
        _write_json(self.config.run.output_dir / "summary.json", summary)
        _write_json(self.config.run.output_dir / "resolved_config.json", self.config.raw_config)
        return summary

    def run_single_task(self, task_id: str) -> Dict[str, Any]:
        env = self._create_env(task_id)
        task_started_at = time.time()
        trace: List[Dict[str, Any]] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            init_state = env.get_init_state()
            messages = copy.deepcopy(init_state.get("state", []))
            tools = self._current_tools(env)
            terminated = False
            score = 0.0
            error_message = ""

            for turn_index in range(1, self.config.run.max_turns + 1):
                model_result = self.client.create_chat_completion(messages=messages, tools=tools)
                assistant_message = model_result["assistant_message"]
                env_action = _assistant_message_for_history(assistant_message)
                usage = model_result["usage"]
                for key in total_usage:
                    total_usage[key] += usage.get(key, 0)

                step_result = env.step(copy.deepcopy(env_action))
                env_state = step_result.get("state", [{}])[0]
                terminated = bool(step_result.get("is_terminated", False))
                score = float(step_result.get("reward", 0.0) or 0.0)

                messages.append(env_action)
                messages.append({"role": "user", "content": env_state.get("content", "")})
                if not terminated:
                    tools = self._current_tools(env)

                trace.append(
                    {
                        "turn": turn_index,
                        "assistant_raw": assistant_message,
                        "assistant_env_action": env_action,
                        "env_state": env_state,
                        "finish_reason": model_result.get("finish_reason"),
                        "usage": usage,
                        "reward": score if terminated else 0.0,
                        "terminated": terminated,
                    }
                )

                if terminated:
                    break

            if not terminated:
                score = float(env.evaluate(params={"sparse": True}) or 0.0)
                error_message = f"max_turns={self.config.run.max_turns} reached before completion"

            result = {
                "task_id": task_id,
                "status": "ok" if terminated else "max_turns",
                "score": score,
                "terminated": terminated,
                "turns_completed": len(trace),
                "duration_sec": round(time.time() - task_started_at, 3),
                "usage": total_usage,
                "error": error_message,
            }
            if self.config.run.save_trace:
                trace_path = self.config.run.output_dir / "traces" / f"{task_id}.json"
                _write_json(trace_path, {"task_id": task_id, "trace": trace, "result": result})
                result["trace_path"] = str(trace_path)
            return result

        except Exception as exc:
            result = {
                "task_id": task_id,
                "status": "error",
                "score": 0.0,
                "terminated": False,
                "turns_completed": len(trace),
                "duration_sec": round(time.time() - task_started_at, 3),
                "usage": total_usage,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            if self.config.run.save_trace:
                trace_path = self.config.run.output_dir / "traces" / f"{task_id}.json"
                _write_json(trace_path, {"task_id": task_id, "trace": trace, "result": result})
                result["trace_path"] = str(trace_path)
            return result
        finally:
            try:
                env.close()
            except Exception:
                pass

    def _build_summary(self, task_ids: List[str], started_at: float) -> Dict[str, Any]:
        scores = [float(item.get("score", 0.0) or 0.0) for item in self.results]
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        status_counter: Dict[str, int] = {}

        for item in self.results:
            status = item.get("status", "unknown")
            status_counter[status] = status_counter.get(status, 0) + 1
            usage = item.get("usage") or {}
            for key in total_usage:
                total_usage[key] += int(usage.get(key, 0) or 0)

        average_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "model": self.config.model.model,
            "base_url": self.config.model.base_url,
            "task_count": len(task_ids),
            "average_score": average_score,
            "status_counter": status_counter,
            "duration_sec": round(time.time() - started_at, 3),
            "usage": total_usage,
            "tasks": [item["task_id"] for item in self.results],
        }


def _load_runner_config(args: argparse.Namespace) -> RunnerConfig:
    config_path = Path(args.config).resolve() if args.config else None
    raw: Dict[str, Any]
    if config_path:
        raw = _read_json(config_path)
        config_base_dir = config_path.parent
    else:
        raw = {
            "model": {
                "base_url": args.base_url,
                "api_key": args.api_key,
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "timeout": args.timeout,
                "use_tools_api": not args.disable_tools_api,
            },
            "bfcl": {
                "benchmark": args.benchmark,
                "data_path": args.data_path,
                "answer_path": args.answer_path,
                "split_ids_path": args.split_ids_path,
                "split": args.split,
                "task_ids": args.task_ids.split(",") if args.task_ids else [],
                "task_prefix": args.task_prefix,
                "task_contains": args.task_contains,
                "max_tasks": args.max_tasks,
                "is_open_query": args.is_open_query,
            },
            "run": {
                "output_dir": args.output_dir,
                "max_turns": args.max_turns,
                "stop_on_error": args.stop_on_error,
                "save_trace": not args.no_save_trace,
                "parallelism": args.parallelism,
            },
        }
        raw = _expand_env_vars(raw)
        config_base_dir = REPO_ROOT

    model_raw = raw.get("model") or {}
    bfcl_raw = raw.get("bfcl") or {}
    run_raw = raw.get("run") or {}
    preset_bfcl_raw = _resolve_benchmark_preset(bfcl_raw.get("benchmark"))
    merged_bfcl_raw = {**preset_bfcl_raw, **{key: value for key, value in bfcl_raw.items() if value is not None}}
    if not merged_bfcl_raw.get("data_path"):
        merged_bfcl_raw["data_path"] = "env_service/environments/bfcl/bfcl_data/multi_turn_processed.jsonl"
    if not merged_bfcl_raw.get("answer_path"):
        merged_bfcl_raw["answer_path"] = "env_service/environments/bfcl/bfcl_eval/possible_answer"
    if not merged_bfcl_raw.get("split_ids_path"):
        merged_bfcl_raw["split_ids_path"] = "env_service/environments/bfcl/bfcl_data/multi_turn_split_ids.json"
    if merged_bfcl_raw.get("split") is None:
        merged_bfcl_raw["split"] = "test"

    model_config = ModelConfig(
        base_url=model_raw["base_url"],
        api_key=model_raw.get("api_key") or os.getenv("OPENAI_API_KEY", ""),
        model=model_raw["model"],
        temperature=float(model_raw.get("temperature", 0.0)),
        max_tokens=int(model_raw.get("max_tokens", 2048)),
        timeout=int(model_raw.get("timeout", 180)),
        use_tools_api=bool(model_raw.get("use_tools_api", True)),
        extra_body=model_raw.get("extra_body") or {},
        headers=model_raw.get("headers") or {},
    )

    bfcl_config = BfclConfig(
        data_path=_resolve_path(config_base_dir, merged_bfcl_raw["data_path"], required=True),
        answer_path=_resolve_path(config_base_dir, merged_bfcl_raw["answer_path"], required=True),
        split_ids_path=_resolve_path(config_base_dir, merged_bfcl_raw["split_ids_path"], required=True),
        benchmark=merged_bfcl_raw.get("benchmark"),
        split=str(merged_bfcl_raw.get("split", "test")),
        task_ids=[str(task_id) for task_id in merged_bfcl_raw.get("task_ids", [])],
        task_prefix=merged_bfcl_raw.get("task_prefix"),
        task_contains=merged_bfcl_raw.get("task_contains"),
        max_tasks=int(merged_bfcl_raw["max_tasks"]) if merged_bfcl_raw.get("max_tasks") is not None else None,
        is_open_query=bool(merged_bfcl_raw.get("is_open_query", False)),
    )

    output_dir = _resolve_path(config_base_dir, run_raw["output_dir"], required=False)
    if output_dir is None:
        raise ValueError("run.output_dir is required")

    run_config = RunConfig(
        output_dir=output_dir,
        max_turns=int(run_raw.get("max_turns", 24)),
        stop_on_error=bool(run_raw.get("stop_on_error", False)),
        save_trace=bool(run_raw.get("save_trace", True)),
        parallelism=max(1, int(run_raw.get("parallelism", 1))),
    )

    return RunnerConfig(
        model=model_config,
        bfcl=bfcl_config,
        run=run_config,
        raw_config=_mask_sensitive_fields(raw),
    )


def _mask_sensitive_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    masked = copy.deepcopy(raw)
    if isinstance(masked.get("model"), dict) and masked["model"].get("api_key"):
        masked["model"]["api_key"] = "***"
    return masked


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone BFCL evaluator for OpenAI-compatible model APIs."
    )
    parser.add_argument("--config", help="Path to a JSON config file.")

    parser.add_argument("--base-url", default="", help="Model API base URL, for example http://host:8000/v1")
    parser.add_argument("--api-key", default="", help="Model API key. Defaults to OPENAI_API_KEY when omitted.")
    parser.add_argument("--model", default="", help="Model name passed to /chat/completions")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--disable-tools-api", action="store_true", help="Do not pass tools to the API. The model must emit <tool_call> tags in plain text.")

    parser.add_argument("--benchmark", default=None, help="High-level BFCL benchmark selector, such as multi_turn, multi_turn_base, multi_turn_miss_func, multi_turn_miss_param, multi_turn_long_context, single_turn, live, non_live, live_simple, live_multiple, live_parallel, live_parallel_multiple, live_irrelevance, live_relevance, simple, simple_python, simple_java, simple_javascript, multiple, parallel, parallel_multiple, irrelevance.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--answer-path", default=None)
    parser.add_argument("--split-ids-path", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--task-ids", default="", help="Comma-separated task ids. When set, split selection is ignored.")
    parser.add_argument("--task-prefix", default=None, help="Keep only task ids starting with this prefix.")
    parser.add_argument("--task-contains", default=None, help="Keep only task ids containing this text.")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--is-open-query", action="store_true", help="Pass is_open_query=True to BFCL env.")

    parser.add_argument("--output-dir", default="bfcl_runner/output/latest")
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-save-trace", action="store_true")
    parser.add_argument("--parallelism", type=int, default=1, help="Number of worker threads for concurrent task evaluation.")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    config = _load_runner_config(args)
    runner = BfclEvaluatorRunner(config)
    summary = runner.run()
    print(_json_dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
