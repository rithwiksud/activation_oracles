"""MCP eval server (Harbor sidecar): the agent's only way to score an oracle. The eval code, target LoRAs and prompts
live only in this container. The agent copies an adapter into the shared volume and calls a tool; it gets back
aggregate accuracies only.

- evaluate_val: the validation metrics (Taboo, and the AO paper's 7 OOD classification datasets).
"""

import json
import os
import re
import subprocess
import tempfile
import threading
import time

from mcp.server.fastmcp import FastMCP

SHARED = "/shared"
# Collected as a trial artifact (see task.toml), so failed eval calls can be diagnosed after the trial
LOG_DIR = "/var/log/ao_eval"
os.makedirs(LOG_DIR, exist_ok=True)
EVAL_DIR = "/opt/ao_eval"
ENV = {**os.environ, "PYTHONPATH": f"{EVAL_DIR}:/opt/ao_ref"}
EVALS = {
    "taboo": [f"{EVAL_DIR}/eval_taboo.py"],
    "ood_classification": [f"{EVAL_DIR}/eval_classification.py", "--data-dir", f"{EVAL_DIR}/classification_test"],
}
BUDGETS = {"val": int(os.environ["AO_EVAL_MAX_VAL_CALLS"])}
# Smoke tests score a handful of items per eval (set only by the smoke-test task)
if os.environ["AO_EVAL_SMOKE"] == "1":
    EVALS["taboo"] += ["--max-words", "2", "--max-prompts", "3"]
    EVALS["ood_classification"] += ["--max-items", "16"]

server = FastMCP("ao-eval", host="0.0.0.0", port=8000)
lock = threading.Lock()  # one eval at a time: they share the GPU with training
calls_used = {"val": 0}


def log_call(kind: str, adapter_name: str, outcome: str) -> None:
    with open(f"{LOG_DIR}/calls.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {kind} {adapter_name!r} {outcome}\n")


def run_evals(kind: str, adapter_name: str, tasks: list[str]) -> dict:
    try:
        result = _run_evals(kind, adapter_name, tasks)
    except Exception as e:
        log_call(kind, adapter_name, f"ERROR {type(e).__name__}: {e}")
        raise
    log_call(kind, adapter_name, json.dumps(result))
    return result


def _run_evals(kind: str, adapter_name: str, tasks: list[str]) -> dict:
    assert re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", adapter_name) and adapter_name not in (".", ".."), "invalid adapter_name"
    path = os.path.join(SHARED, adapter_name)
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        assert os.path.isfile(os.path.join(path, f)), f"missing {f} in /shared/{adapter_name}"
    with lock:
        assert calls_used[kind] < BUDGETS[kind], f"{kind} eval budget exhausted ({BUDGETS[kind]} calls used)"
        calls_used[kind] += 1
        scores = {}
        with tempfile.TemporaryDirectory() as tmp:
            for task in tasks:
                out = os.path.join(tmp, f"{task}.json")
                # Separate processes, so GPU memory is freed after each eval. Logs stay in this container.
                log_path = f"{LOG_DIR}/{kind}{calls_used[kind]}_{task}.log"
                with open(log_path, "w") as log:
                    proc = subprocess.run(["python", *EVALS[task], "--oracle-lora", path, "--output", out], env=ENV,
                                          stdout=log, stderr=log)
                if proc.returncode != 0:
                    # The GPU is shared with the agent's training. Running out of memory is not the adapter's fault,
                    # so it doesn't use up a call.
                    if "torch.OutOfMemoryError" in open(log_path).read():
                        calls_used[kind] -= 1
                        raise RuntimeError(
                            "Eval ran out of GPU memory: it needs about 12 GB free, and your other GPU processes left "
                            "too little. This call was not counted. Free some GPU memory and call again."
                        )
                    raise subprocess.CalledProcessError(proc.returncode, proc.args)
                scores[task] = round(json.load(open(out))["summary"]["accuracy"], 4)
        return {**scores, "calls_remaining": BUDGETS[kind] - calls_used[kind]}


@server.tool()
def evaluate_val(adapter_name: str) -> dict:
    """Score an activation oracle LoRA adapter for Qwen/Qwen3.5-0.8B on the validation metrics.

    First copy the adapter directory (adapter_config.json + adapter_model.safetensors) to /shared/<adapter_name>.
    Returns {"taboo": accuracy, "ood_classification": accuracy (pooled over 7 held-out yes/no classification datasets),
    "calls_remaining": n}. Takes about 5-7 minutes.
    """
    return run_evals("val", adapter_name, ["taboo", "ood_classification"])


if __name__ == "__main__":
    server.run(transport="streamable-http")
