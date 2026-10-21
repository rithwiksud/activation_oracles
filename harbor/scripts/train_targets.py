"""Train the target LoRAs that hold hidden information (PersonaQA personas, Taboo secret words) for Qwen3.5-0.8B.

Each adapter is trained once and skipped if it already exists. The recipes mirror the __main__ blocks of
nl_probes/trl_training/{personaqa,taboo}_train.py (Taboo uses the ultrachat "_50_mix" variant, as in the Qwen3-8B
targets used by experiments/taboo_open_ended_eval.py).

Usage (from the repo root): python harbor/scripts/train_targets.py <output_dir> [personaqa|taboo|taboo_reversed|all] <personaqa_dir>

<personaqa_dir> is the extracted datasets/personaqa_data.zip "shuffled" folder (bios + interviews). It is kept out of
the repo's datasets/ so the training documents never enter the agent's image.
"""

import sys
from pathlib import Path

sys.path.insert(0, "nl_probes/trl_training")

import personaqa_train
import taboo_train
from config import CustomSFTConfig, EvalConfig
from datasets import load_dataset
from transformers import AutoTokenizer, EarlyStoppingCallback

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
TABOO_WORDS = [
    "ship", "wave", "song", "snow", "rock", "moon", "jump", "green", "flame", "flag",
    "dance", "cloud", "clock", "chair", "salt", "book", "blue", "gold", "leaf", "smile",
]  # fmt: skip
BATCH_SIZE = 8
REAL_BATCH_SIZE = 8

out_dir = Path(sys.argv[1])
which = sys.argv[2]
personaqa_dir = sys.argv[3]
# taboo_reversed walks the words in reverse, so a second process can train in parallel with a "taboo" one
assert which in {"personaqa", "taboo", "taboo_reversed", "all"}, which


def train_personaqa(lora_path: Path) -> None:
    config = EvalConfig(model_name=MODEL_NAME, model_lora_dir=str(out_dir))
    sft_config = CustomSFTConfig(model_name=MODEL_NAME, batch_size=BATCH_SIZE, real_batch_size=REAL_BATCH_SIZE)
    sft_config.run_name = "personaqa_3_epochs"
    sft_config.num_train_epochs = 3
    sft_config.completion_only_loss = True

    ds = personaqa_train.create_personaqa_dataset(personaqa_dir)
    train_size = int(len(ds) * 0.99)
    eval_size = int(len(ds) * 0.01)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = personaqa_train.prepare_sft_dataset(ds.select(range(train_size)), tokenizer)
    eval_ds = personaqa_train.prepare_sft_dataset(ds.select(range(train_size, train_size + eval_size)), tokenizer)

    eval_frequency = len(train_ds) // (REAL_BATCH_SIZE * 2)
    sft_config.eval_steps = eval_frequency
    sft_config.save_steps = eval_frequency
    personaqa_train.train_with_sft_only(
        train_ds, eval_ds, config.wandb_project, config, sft_config, callbacks=[], save_lora_path=lora_path
    )


def train_taboo(word: str, lora_path: Path) -> None:
    config = EvalConfig(model_name=MODEL_NAME, model_lora_dir=str(out_dir))
    sft_config = CustomSFTConfig(model_name=MODEL_NAME, batch_size=BATCH_SIZE, real_batch_size=REAL_BATCH_SIZE)
    sft_config.run_name = f"taboo_{word}"
    sft_config.num_train_epochs = 10.0

    ds = taboo_train.create_incremental_turn_dataset(load_dataset(f"bcywinski/taboo-{word}", split="train"))
    train_size = int(len(ds) * 0.9)
    eval_size = int(len(ds) * 0.1)
    raw_train_ds = ds.select(range(train_size))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = taboo_train.prepare_sft_dataset(raw_train_ds, tokenizer, final_message_loss_only=True)
    eval_ds = taboo_train.prepare_sft_dataset(
        ds.select(range(train_size, train_size + eval_size)), tokenizer, final_message_loss_only=True
    )
    train_ds = taboo_train.combine_with_ultrachat(
        raw_train_ds=raw_train_ds,
        tokenized_train_ds=train_ds,
        chat_dataset_name="HuggingFaceH4/ultrachat_200k",
        tokenizer=tokenizer,
        random_seed=config.random_seed,
        final_message_loss_only=True,
    )

    eval_frequency = len(train_ds) // (REAL_BATCH_SIZE * 2)
    sft_config.eval_steps = eval_frequency
    sft_config.save_steps = eval_frequency
    taboo_train.train_with_sft_only(
        train_ds,
        eval_ds,
        config.wandb_project,
        config,
        sft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        save_lora_path=lora_path,
    )


if which in {"personaqa", "all"}:
    lora_path = out_dir / "personaqa"
    if lora_path.exists():
        print(f"{lora_path} exists, skipping")
    else:
        train_personaqa(lora_path)

if which in {"taboo", "taboo_reversed", "all"}:
    for word in reversed(TABOO_WORDS) if which == "taboo_reversed" else TABOO_WORDS:
        lora_path = out_dir / "taboo" / word
        if lora_path.exists():
            print(f"{lora_path} exists, skipping")
        else:
            train_taboo(word, lora_path)
