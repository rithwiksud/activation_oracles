"""Remove check-task eval data from the staged agent-visible repo, so evaluation is only possible through the MCP eval
server: the PersonaQA personas, and the three test-only OOD classification sets that nl_probes/sft.py otherwise
scores during training (ag_news, language_identification, singular_plural).

Usage: python strip_eval_data.py <staged ao_ref dir>
"""

import shutil
import sys
from pathlib import Path

ref = Path(sys.argv[1])
sft = ref / "nl_probes" / "sft.py"
s = sft.read_text()
for entry in [
    '        "ag_news": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["test"]},\n',
    '        "language_identification": {\n'
    '            "num_train": main_train_size,\n'
    '            "num_test": main_test_size,\n'
    '            "splits": ["test"],\n'
    '            # language identification has very long sequence lengths\n'
    '            "batch_size": 4,\n'
    '        },\n',
    '        "singular_plural": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},\n',
]:
    assert s.count(entry) == 1, entry
    s = s.replace(entry, "")
sft.write_text(s)

for d in ["datasets/personaqa_data", "datasets/classification_datasets/ag_news", "datasets/classification_datasets/singular_plural"]:
    shutil.rmtree(ref / d)
print(f"stripped check-task eval data from {ref}")
