# Verifier: retrains the submission and scores it on the held-out tests plus the train/check tasks. Eval code, targets
# and data are readable only by root; the submitted train.sh runs as the unprivileged `trainer` user.
FROM ao-base:latest

COPY ao_targets/ /opt/ao_targets/
COPY ao_eval/ /opt/ao_eval/
COPY ao_heldout/ /opt/ao_heldout/
COPY tests/ /tests/
RUN chmod -R go-rwx /opt/ao_targets /opt/ao_eval /opt/ao_heldout /tests && useradd -m trainer
