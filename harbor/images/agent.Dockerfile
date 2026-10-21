# Agent sandbox: a git-tracked working copy of the reference repo, with Claude Code preinstalled so agent setup
# needs no network.
FROM ao-base:latest

RUN curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash \
    && echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.bashrc
ENV PATH=/root/.local/bin:$PATH

# Managed settings take precedence over Harbor's CLAUDE_CONFIG_DIR and can't be overridden by the agent.
# Claude Code ends a turn after 9 consecutive Stop-hook blocks by default, which would end a headless session early.
ENV CLAUDE_CODE_STOP_HOOK_BLOCK_CAP=100000
# An eval MCP call takes several minutes (longer while training shares the GPU)
ENV MCP_TOOL_TIMEOUT=1200000
COPY agent_files/ /opt/ao_agent/
RUN chmod +x /opt/ao_agent/*.sh /opt/ao_agent/time_left \
    && ln -s /opt/ao_agent/time_left /usr/local/bin/time_left \
    && mkdir -p /etc/claude-code && mv /opt/ao_agent/managed-settings.json /etc/claude-code/managed-settings.json

RUN mkdir -p /app && cp -r /opt/ao_ref /app/repo \
    && cd /app/repo \
    && git init -q && git -c user.name=baseline -c user.email=baseline@local add -A \
    && git -c user.name=baseline -c user.email=baseline@local commit -qm "Baseline"
WORKDIR /app/repo
# nl_probes is imported from the working copy (the image installs dependencies only)
ENV PYTHONPATH=/app/repo
