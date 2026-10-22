"""Build the classification test sets the Harbor evals score: the AO paper's 7 OOD datasets and test-set config (the
config of experiments/classification_eval.py at the paper's final classification eval, commit 0606a87: layers 25/50/75%,
3 questions per sample, 250 test samples per dataset), except that the oracle reads the full sequence (every context
token) instead of the paper's single token, like every Harbor eval.

Usage: python build_classification_test_sets.py <out dir>
"""

import sys

from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.classification import ClassificationDatasetConfig, ClassificationDatasetLoader

out_dir = sys.argv[1]
MODEL_NAME = "Qwen/Qwen3.5-0.8B"

# The paper's 7 held-out (OOD) datasets (App. C.3)
OOD_DATASETS = [
    "ag_news",
    "language_identification",
    "singular_plural",
    "engels_headline_istrump",
    "engels_headline_isobama",
    "engels_headline_ischina",
    "engels_hist_fig_ismale",
]
for name in OOD_DATASETS:
    loader = ClassificationDatasetLoader(
        dataset_config=DatasetLoaderConfig(
            custom_dataset_params=ClassificationDatasetConfig(
                classification_dataset_name=name,
                # Window ends at the last token and is longer than any context, so it covers the full sequence
                max_end_offset=-1,
                min_end_offset=-1,
                max_window_size=100_000,
                min_window_size=100_000,
            ),
            num_train=0,
            num_test=250,
            splits=["test"],
            model_name=MODEL_NAME,
            layer_percents=[25, 50, 75],
            save_acts=False,
            batch_size=32 if name == "language_identification" else 256,
            dataset_folder=out_dir,
        )
    )
    data = loader.load_dataset("test")
    assert all(dp.context_positions == list(range(len(dp.context_input_ids))) for dp in data), name
    print(f"{name}: {len(data)} test datapoints")
