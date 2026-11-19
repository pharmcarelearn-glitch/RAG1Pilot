# Running rag1pilot on a small VM (self-hosted worker)

`agents serve` runs the agent as a worker on a machine you control while Mistral
stores the sessions, traces, and routes chats by agent name. It consumes **no
managed-deployment slot** (useful when the organization's managed-deployment
limit is reached), needs **no inbound ports**, and survives reboots through
systemd.

## What you need

- A small VM: 1 vCPU / 1–2 GB RAM is enough (AWS t3.small, GCP e2-small, Hetzner
  CX22, a $5 droplet…), Ubuntu 22.04+ or Debian 12+.
- SSH access to it.
- A Mistral API key whose user has a workspace role with Workflows access
  (Developer or Workspace Contributor — a workspace admin assigns it in the
  [admin console](https://admin.mistral.ai): Organization > Workspaces > your
  workspace > Members).
- Outbound HTTPS to `api.mistral.ai` (default on any VM; no inbound rules to add).

## Install

On the VM:

```bash
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/pharmcarelearn-glitch/RAG1Pilot.git /opt/rag1pilot
cd /opt/rag1pilot
git checkout <the branch with the agent, e.g. main after the PR merges>
sudo MISTRAL_API_KEY="<your-key>" bash scripts/setup-vm.sh
```

The script installs uv, creates a dedicated `rag1pilot` system user, stores the
key in `/etc/rag1pilot-agent/.env` (root-only, mode 600), runs `uv sync --frozen`,
installs the `rag1pilot-agent` systemd unit, and starts and enables the service.

## Verify

```bash
systemctl status rag1pilot-agent
journalctl -u rag1pilot-agent -f          # live logs
journalctl -u rag1pilot-agent --no-pager | grep 'Try the agent'
```

The last command prints the worker banner, including the **Vibe Code Web** chat
link for the served agent (`https://chat.mistral.ai/managed-agents/rag1pilot?...`).
Restarting the VM restarts the worker automatically; Mistral keeps finished
session steps, so nothing is lost across restarts.

## Talk to it

- **Vibe Code Web:** open the "Try the agent" link from the logs and chat; the
  sessions run on the VM, so it must stay on.
- **From your laptop:** install `uv`, put a `MISTRAL_API_KEY` for the same
  workspace in your shell, and run `agents chat rag1pilot` — `agents chat`
  without a project name would run a local copy, so always pass the agent name.
- **Through a Vibe session:** ask the assistant to run
  `agents chat rag1pilot -m "<your question>"` with the key in `.env`.
- **From your app:** `agents.Agent.load_from(agent_name="rag1pilot")`.

First use asks you to connect Google Drive in the chat; the connector then acts
with the Drive permissions of the user who talks to the agent.

## Update the agent

```bash
cd /opt/rag1pilot
sudo -u rag1pilot git pull
sudo systemctl restart rag1pilot-agent
```

## Notes and limits

- The worker only answers while the VM runs. Stop it with
  `sudo systemctl stop rag1pilot-agent`; sessions and history stay on Mistral.
- Self-hosted sessions end after 4 h idle or 24 h total by default; tune with
  `AGENTS_SDK_IDLE_TIMEOUT_SECONDS` / `AGENTS_SDK_LIFETIME_LIMIT_SECONDS` in
  `/etc/rag1pilot-agent/.env` (see
  [session limits](https://docs.mistral.ai/studio/agents/learn/where-it-runs/session-lifecycle)).
- Anytime later, the same project deploys fully managed with
  `uv run --env-file .env agents deploy` (needs a free managed-deployment slot).
- Troubleshooting: run `AGENTS_SDK_DEBUG=1` on a manual
  `uv run --frozen --env-file /etc/rag1pilot-agent/.env agents serve` in
  `/opt/rag1pilot` for full logs; see the
  [`agents serve` docs](https://docs.mistral.ai/studio/agents/learn/where-it-runs/serve).
