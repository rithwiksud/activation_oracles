"""Cap the training data in the Harbor images at the baseline recipe's 10x-reduced mixture, so more data does not exist
in the sandbox (the agent is offline). Keeps exactly what the baseline in nl_probes/sft.py consumes:

- PastLens: the leading FineWeb / LMSYS-chat rows the baseline reads (every variant restarts the stream at row 0).
- LatentQA: the 6,500 (dialog, QA) pairs the baseline samples (the unused system-prompt and eval splits are removed), rewritten with one unique label per pair so the loader
  (which crosses every dialog of a label with every QA of that label) cannot expand them into more.
- Classification: for each training dataset and each num_qa_per_sample variant, the baseline's train/test samples as a
  fixed pool. get_classification_datapoints is patched to read the pools, and the raw source files are removed.

Run on the staged copies from build_images.sh (after strip_eval_data.py), with the host venv (it needs the HF caches
of the classification datasets once, to build the pools).

Usage: python cap_training_data.py <staged ao_ref dir> <staged ao_data dir>
"""

import json
import math
import os
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ref = Path(sys.argv[1]).resolve()
data = Path(sys.argv[2]).resolve()

# The baseline recipe this cap is computed for. Fail loudly if sft.py no longer matches.
NUM_PASTLENS = 10_000
BATCH_SIZE = 256
LATENTQA_NUM_TRAIN = 6_500
CLS_TRAIN, CLS_TEST = 600, 250
CLS_DATASETS = ["geometry_of_truth", "relations", "sst2", "md_gender", "snli", "ner", "tense"]
CLS_NUM_QA = [2, 1]
SEED = 42
sft = (ref / "nl_probes" / "sft.py").read_text()
for line in [
    "    num_datapoints = 10_000  # 10x scaled down from 100_000\n",
    "            train_batch_size = 256\n",
    "            num_train=6_500,  # 10x scaled down",
    "    main_train_size = 600  # 10x scaled down from 6000\n",
    "    main_test_size = 250\n",
    "            num_qa_per_sample=2,\n",
    "            num_qa_per_sample=1,\n",
]:
    assert line in sft, line
assert "seed: int = 42" in (ref / "nl_probes" / "dataset_classes" / "act_dataset_manager.py").read_text()

# Import the staged code (identical to the repo's apart from the local PastLens paths) from inside the staged tree,
# since the loaders read data through relative paths
os.chdir(ref)
sys.path.insert(0, str(ref))
import nl_probes.dataset_classes.classification as classification  # noqa: E402
import nl_probes.dataset_classes.classification_dataset_manager as cdm  # noqa: E402
import nl_probes.dataset_classes.misc.latentqa_loader as latentqa_loader  # noqa: E402
import nl_probes.dataset_classes.past_lens_dataset as past_lens_dataset  # noqa: E402
from nl_probes.utils.common import load_tokenizer  # noqa: E402

# ---- PastLens: count the rows the baseline consumes, keep only those ----
rows_read: dict[str, int] = {}
real_load_dataset = past_lens_dataset.load_dataset


def counting_load_dataset(path, **kwargs):
    def rows():
        for i, row in enumerate(real_load_dataset(path, **kwargs)):
            rows_read[path] = i + 1
            yield row

    return rows()


past_lens_dataset.load_dataset = counting_load_dataset
gen = past_lens_dataset.hf_mixed_dataset_to_generator(
    load_tokenizer("Qwen/Qwen3.5-0.8B"), str(data / "fineweb"), str(data / "lmsys-chat-1m")
)
# collect_past_lens_acts draws whole batches
for _ in range(math.ceil(NUM_PASTLENS / BATCH_SIZE) * BATCH_SIZE):
    next(gen)
past_lens_dataset.load_dataset = real_load_dataset
for name in ["fineweb", "lmsys-chat-1m"]:
    path = data / name / "train.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    n = rows_read[str(data / name)]
    assert n < len(lines), f"{name}: baseline reads {n} rows but only {len(lines)} were dumped"
    path.write_text("".join(lines[:n]))
    print(f"PastLens {name}: kept {n} of {len(lines)} rows")

# ---- LatentQA: keep the baseline's sampled pairs, one unique label per pair ----
lqa_dir = ref / "datasets" / "latentqa_datasets"
paths = latentqa_loader.DataPaths(
    system=None,
    stimulus_completion="datasets/latentqa_datasets/train/stimulus_completion.json",
    stimulus="datasets/latentqa_datasets/train/stimulus.json",
    control="datasets/latentqa_datasets/train/control.json",
    qa="datasets/latentqa_datasets/train/qa.json",
)
ds = latentqa_loader.load_latentqa_dataset(paths, filter_prefixes=[], train_percent=1.0, seed=SEED)
# Same selection as LatentQADatasetLoader.create_dataset
selected = random.Random(SEED).sample(list(range(len(ds))), LATENTQA_NUM_TRAIN)
fields = ["system", "control_user", "control_thought", "control_model", "stimulus_user", "stimulus_thought", "stimulus_model"]
items: dict[str, list[dict]] = {"stimulus_completion": [], "stimulus": [], "control": []}
qa: dict[str, list[list[str]]] = {}
for k, idx in enumerate(selected):
    g, label_idx, data_idx, qa_idx = ds.index[idx]
    source = ds.SOURCES[g]
    label = ds.labels_per_group[g][label_idx]
    behavior = ds.data_groups[g][label][data_idx]
    new_label = f"{label}-{k}"
    items[source].append({**{f: v for f, v in zip(fields, behavior) if v != ""}, "label": new_label})
    qa[new_label] = [ds.qa_data[label][qa_idx]]
for source, source_items in items.items():
    (lqa_dir / "train" / f"{source}.json").write_text(json.dumps(source_items))
(lqa_dir / "train" / "qa.json").write_text(json.dumps(qa))
shutil.rmtree(lqa_dir / "eval")
# The system-prompt split is not part of the baseline mixture (paths.system is None)
(lqa_dir / "train" / "system.json").unlink()
check = latentqa_loader.load_latentqa_dataset(paths, filter_prefixes=[], train_percent=1.0, seed=SEED)
assert len(check) == LATENTQA_NUM_TRAIN, len(check)
print(f"LatentQA: kept {len(check)} of {len(ds)} pairs ({ {s: len(v) for s, v in items.items()} })")

# ---- Classification: fixed train/test pools per dataset and num_qa_per_sample ----
cls_dir = ref / "datasets" / "classification_datasets"
pools = {}
for name in CLS_DATASETS:
    for num_qa in CLS_NUM_QA:
        # Same selection as the original get_classification_datapoints
        classification.set_seed(SEED)
        samples = cdm.get_samples_from_groups([name], num_qa)
        random.shuffle(samples)
        assert len(samples) >= CLS_TRAIN + CLS_TEST, name
        pools[f"{name}_qa{num_qa}"] = {
            "train": [asdict(s) for s in samples[:CLS_TRAIN]],
            "test": [asdict(s) for s in samples[-CLS_TEST:]],
        }
# Remove every raw data file, but keep the directory tree (the relations loader lists its folder at import) and the
# question paraphrases
for p in sorted(cls_dir.rglob("*")):
    if p.is_file() and "paraphrases" not in p.parts:
        p.unlink()
(cls_dir / "pools").mkdir()
for key, pool in pools.items():
    (cls_dir / "pools" / f"{key}.json").write_text(json.dumps(pool))
print(f"Classification: wrote {len(pools)} pools of {CLS_TRAIN} train + {CLS_TEST} test samples")

cls_py = ref / "nl_probes" / "dataset_classes" / "classification.py"
s = cls_py.read_text()
old = """    set_seed(random_seed)
    all_examples = classification_dataset_manager.get_samples_from_groups(
        [dataset_name],
        num_qa_per_sample,
    )

    random.shuffle(all_examples)

    assert len(all_examples) >= train_examples + test_examples, "Not enough examples to split"
    train_examples = all_examples[:train_examples]
    test_examples = all_examples[-test_examples:]
"""
new = """    set_seed(random_seed)
    # This environment's training data is capped: each dataset ships a fixed pool of train and test samples per
    # num_qa_per_sample (the baseline's 10x-reduced mixture). The raw source data is not available.
    with open(f"datasets/classification_datasets/pools/{dataset_name}_qa{num_qa_per_sample}.json") as f:
        pool = json.load(f)
    assert train_examples <= len(pool["train"]), f"only {len(pool['train'])} train samples exist for {dataset_name}"
    assert test_examples <= len(pool["test"]), f"only {len(pool['test'])} test samples exist for {dataset_name}"
    train_examples = [_pool_sample(s) for s in pool["train"][:train_examples]]
    test_examples = [_pool_sample(s) for s in pool["test"][len(pool["test"]) - test_examples :]]
"""
helper = """

def _pool_sample(fields: dict) -> classification_dataset_manager.ContextQASample:
    # Pool samples are finished (get_samples_from_groups already stripped the "# " question prefix), so skip the
    # constructor's checks on raw loader output
    sample = object.__new__(classification_dataset_manager.ContextQASample)
    sample.__dict__.update(fields)
    return sample
"""
assert s.count(old) == 1
s = s.replace(old, new).replace("import random\n", "import json\nimport random\n", 1)
assert s.count("\ndef get_classification_datapoints(") == 1
s = s.replace("\ndef get_classification_datapoints(", helper + "\n\ndef get_classification_datapoints(")
cls_py.write_text(s)
print("patched get_classification_datapoints to read the pools")
