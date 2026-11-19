# rag1pilot

Conversational data analyst over CSV files stored in Google Drive, built with the
[Mistral Agents SDK](https://docs.mistral.ai/studio/agents/getting-started/quickstart)
and deployed on Mistral-managed infrastructure.

You give it a link to a CSV in your Drive and ask questions in plain language.
It answers with real numbers computed from your data through a small set of
fixed, typed tools — there is no arbitrary code execution on the data.

## Tools

| Tool | Purpose |
| --- | --- |
| `load_csv` | Registers a CSV (fetched via the `google_drive_mcp` connector) as the working dataset; returns a summary only |
| `list_loaded_datasets` | Lists datasets loaded in the current session |
| `describe_columns` | Per-column dtype, missing, distinct, numeric stats / top values |
| `filter_rows` | AND-combined fixed operators (eq, ne, gt, ge, lt, le, contains, startswith, endswith, in, not_in, is_null, not_null) with sort and row limit |
| `group_by_aggregate` | Group by one or more columns; count, nunique, sum, mean, median, min, max, std, var, first, last |

File bytes are moved from the Drive connector into `load_csv` inside one
programmatic tool-call step, so large CSV content never enters the model's
context. Loaded datasets live for the whole session, so follow-up questions
work without re-loading or re-explaining anything.

## Local checks

```bash
uv sync --frozen
uv run --frozen pytest
uv run --frozen agents info
```

## Chat locally

Put a Mistral API key in `.env` as `MISTRAL_API_KEY`, then:

```bash
uv run --env-file .env agents chat
```

## Deploy (Mistral-managed)

Commit and push the branch, then:

```bash
uv run --env-file .env agents deploy
```

Chat with the deployed agent by name (sessions run on Mistral and are durable):

```bash
uv run --env-file .env agents chat rag1pilot
```

The agent acts with the Drive connection of the user who starts the session
(on-behalf-of), so the deployment is hardened; `agents deploy` does this by
default.
