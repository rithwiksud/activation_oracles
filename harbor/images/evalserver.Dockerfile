# MCP eval server (Harbor sidecar): the only place the dev/validation eval code, targets and data exist on the agent
# side. The agent reaches it over MCP and gets aggregate scores within per-trial call budgets.
FROM ao-base:latest

RUN uv pip install --python /opt/venv/bin/python mcp==1.30.0 && rm -rf /root/.cache/uv
COPY ao_targets/ /opt/ao_targets/
COPY ao_eval/ /opt/ao_eval/
COPY eval_server/ /opt/eval_server/
ENV AO_EVAL_MAX_VAL_CALLS=30 AO_EVAL_SMOKE=0
EXPOSE 8000
CMD ["python", "/opt/eval_server/server.py"]
