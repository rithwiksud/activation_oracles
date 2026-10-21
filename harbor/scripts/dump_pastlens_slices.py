"""Dump the first N rows of the PastLens source datasets to local jsonl, for offline Harbor sandboxes.

PastLens consumes one row per datapoint, half pretrain and half chat, and each dataset variant restarts the
stream from the first row. 60k rows of each therefore covers the full-size mixture (100k datapoints per variant).
"""

import json
import sys
from pathlib import Path

from datasets import load_dataset

out_dir = Path(sys.argv[1])
num_rows = int(sys.argv[2])

for name, local_name, key in [
    ("HuggingFaceFW/fineweb", "fineweb", "text"),
    ("lmsys/lmsys-chat-1m", "lmsys-chat-1m", "conversation"),
]:
    path = out_dir / local_name / "train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = iter(load_dataset(name, split="train", streaming=True))
    with open(path, "w") as f:
        for _ in range(num_rows):
            f.write(json.dumps({key: next(ds)[key]}) + "\n")
    print(f"Wrote {num_rows} rows of {name} to {path}")
