"""MCP eval server (Harbor sidecar): the agent's only way to score an oracle. The eval code, target LoRAs and prompts
live only in this container. The agent copies an adapter into the shared volume and calls a tool; it gets back
aggregate accuracies only.

- evaluate_dev: the dev metrics the agent hill-climbs on (Taboo, ag_news), generous budget.
- evaluate_val: the validation metrics (PersonaQA, language identification + singular/plural), small budget.
"""

import json
import os
import re
import subprocess
import tempfile
import threading

from mcp.server.fastmcp import FastMCP

SHARED = "/shared"
EVAL_DIR = "/opt/ao_eval"
ENV = {**os.environ, "PYTHONPATH": f"{EVAL_DIR}:/opt/ao_ref"}
CLASSIFICATION = [f"{EVAL_DIR}/eval_classification.py", "--data-dir", f"{EVAL_DIR}/classification_test", "--datasets"]
EVALS = {
    "taboo": [f"{EVAL_DIR}/eval_taboo.py"],
    "ag_news": [*CLASSIFICATION, "ag_news"],
    "personaqa": [f"{EVAL_DIR}/eval_personaqa.py"],
    "ood_classification": [*CLASSIFICATION, "language_identification", "singular_plural"],
}
BUDGETS = {"dev": int(os.environ["AO_EVAL_MAX_DEV_CALLS"]), "val": int(os.environ["AO_EVAL_MAX_VAL_CALLS"])}

server = FastMCP("ao-eval", host="0.0.0.0", port=8000)
lock = threading.Lock()  # one eval at a time: they share the GPU with training
calls_used = {"dev": 0, "val": 0}


def run_evals(kind: str, adapter_name: str, tasks: list[str]) -> dict:
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
                with open(f"/var/log/ao_eval_{kind}{calls_used[kind]}_{task}.log", "w") as log:
                    subprocess.run(["python", *EVALS[task], "--oracle-lora", path, "--output", out], env=ENV,
                                   check=True, stdout=log, stderr=log)
                scores[task] = round(json.load(open(out))["summary"]["accuracy"], 4)
        return {**scores, "calls_remaining": BUDGETS[kind] - calls_used[kind]}


@server.tool()
def evaluate_dev(adapter_name: str) -> dict:
    """Score an activation oracle LoRA adapter for Qwen/Qwen3.5-0.8B on the dev metrics (Taboo, ag_news).

    First copy the adapter directory (adapter_config.json + adapter_model.safetensors) to /shared/<adapter_name>.
    Returns {"taboo": accuracy, "ag_news": accuracy, "calls_remaining": n}. Takes a couple of minutes; generous budget.
    """
    return run_evals("dev", adapter_name, ["taboo", "ag_news"])


@server.tool()
def evaluate_val(adapter_name: str) -> dict:
    """Score an activation oracle LoRA adapter on the validation metrics (PersonaQA, OOD classification).

    First copy the adapter directory to /shared/<adapter_name>. Returns {"personaqa": accuracy, "ood_classification":
    accuracy (language identification + singular/plural), "calls_remaining": n}. Takes a few minutes; small budget, so
    use it to confirm promising changes.
    """
    return run_evals("val", adapter_name, ["personaqa", "ood_classification"])


if __name__ == "__main__":
    server.run(transport="streamable-http")
