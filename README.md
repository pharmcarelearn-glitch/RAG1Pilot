# Rag1Pilot — conversational data analyst over Google Drive CSVs

A [Mistral Agents SDK](https://docs.mistral.ai/studio/agents/getting-started/quickstart) agent that
answers plain-language questions about a CSV file stored in the user's Google Drive, with real
numbers computed from the data.

## Design

- **Fixed toolbox, no code interpreter.** The agent runs with the default no-sandbox environment:
  there is no shell, no file system and no Python/network from the model. Google Drive access
  comes from the **user's connector** (`google_drive_mcp`, declared on the agent, so the
  deployment runs on behalf of the session user). The four analysis tools are
  `model_access="direct"`, so the model calls them directly and programs cannot reimplement
  analysis. The connector requirement forces programmatic tool calling on in SDK 1.1.3; its
  TypeScript catalog exposes only these tools (no shell, no files), and instructions forbid
  using it to compute anything:
  - `load_dataset` — parses a Google Drive link (or file id), fetches the file **server-side
    through the user's Google Drive connector** (`google_drive_mcp` via the Connectors gateway,
    so no Drive token ever enters session history), converts Google Sheets to CSV, and loads it
    with pandas.
  - `describe_columns` — per-column type, nulls, uniqueness and statistics.
  - `filter_rows` — one fixed-condition filter (equals, contains, comparisons, in, between,
    is_null, ...), returns the total match count plus up to 100 rows.
  - `groupby_aggregate` — group by one or more columns and compute fixed aggregates
    (`count, nunique, sum, mean, median, min, max, std`).
- **Multi-turn memory.** The dataset stays selected for the whole session; follow-up questions
  need no link re-explanation. If the worker's in-memory cache is lost (worker restart), the
  tools ask the model to re-run `load_dataset` with the link it still has in history, or reload
  the worker-local disk cache.

## Local development

```bash
echo 'MISTRAL_API_KEY=<your-key>' > .env
uv sync
uv run agents info
uv run --env-file .env agents chat
uv run pytest -q            # offline tool tests, no API key needed
```

## Deploy (Mistral-managed)

```bash
uv run agents deploy
```

Chat with the deployed agent by name: `uv run --env-file .env agents chat rag1pilot`.

## Requirements for users

- The **Google Drive connector** connected in Mistral Studio (Context → Connectors). The agent
  reads files with the account connected there; the CSV must be accessible to that account.
- A CSV file link (Google Sheets work: they are exported to CSV; other formats must be exported
  to CSV first).
