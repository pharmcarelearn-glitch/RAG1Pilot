"""Rag1Pilot: a conversational data analyst for CSV files stored in Google Drive.

The agent runs with no sandbox: there is no shell, no file system and no code
interpreter. Google Drive access comes from the user's connector (the session
user's credentials, resolved by the Connectors gateway). Analysis is computed
exclusively by fixed pandas tools: load a Drive CSV, describe columns, filter
rows, and group-by aggregates. Programmatic tool calling is on only as the
connector requires it; the TypeScript catalog it exposes contains just these
tools. The model only phrases the numbers the tools return.
"""

import base64
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import Field

from mistralai import agents
from mistralai.client import Mistral

CONNECTOR_NAME = "google_drive_mcp"
DOWNLOAD_TOOL = "download_file_content"
METADATA_TOOL = "get_file_metadata"
MAX_CSV_BYTES = 50 * 1024 * 1024
CACHE_DIR = Path(tempfile.gettempdir()) / "rag1pilot-cache"

AGGREGATION_FUNCS = ("count", "nunique", "sum", "mean", "median", "min", "max", "std")


def _mistral_client() -> Mistral:
    """Mistral client with the caller's connector credentials.

    On a managed worker there is no API key: RunAs.AUTO follows the
    on-behalf-of session, so the Google Drive connector resolves the session
    user's credentials. Locally the project's MISTRAL_API_KEY is used.
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if api_key:
        return Mistral(api_key=api_key)
    from mistralai.workflows.client import get_mistral_client
    from mistralai.workflows.core.auth import RunAs

    return get_mistral_client(run_as=RunAs.AUTO)


def _extract_file_id(reference: str) -> str:
    ref = reference.strip()
    patterns = [
        r"/file/d/([A-Za-z0-9_-]{10,})",
        r"/spreadsheets/d/([A-Za-z0-9_-]{10,})",
        r"/document/d/([A-Za-z0-9_-]{10,})",
        r"/presentation/d/([A-Za-z0-9_-]{10,})",
        r"[?&]id=([A-Za-z0-9_-]{10,})",
        r"/d/([A-Za-z0-9_-]{10,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, ref)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{15,}", ref):
        return ref
    raise agents.ToolError(
        "Could not find a Google Drive file id in that reference. Pass the file's Drive link "
        "(e.g. https://drive.google.com/file/d/FILE_ID/view) or its raw file id."
    )


async def _call_drive_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
    client = _mistral_client()
    try:
        async with client.beta.connectors.mcp_client(connector_id_or_name=CONNECTOR_NAME) as session:
            result = await session.call_tool(tool_name, arguments)
    except Exception as exc:
        raise agents.ToolError(
            f"Google Drive call failed: {exc}. Make sure the Google Drive connector is connected "
            "for your account in Mistral Studio (Context -> Connectors)."
        ) from exc
    if getattr(result, "isError", False):
        raise agents.ToolError(f"Google Drive returned an error for {tool_name}: {str(result.content)[:300]}")
    data = getattr(result, "structuredContent", None)
    if isinstance(data, dict):
        return data
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            try:
                parsed = json.loads(text)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _cache_path(file_id: str) -> Path:
    return CACHE_DIR / f"{re.sub(r'[^A-Za-z0-9_-]', '_', file_id)}.csv"


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise agents.ToolError("The file is not decodable text; save it as CSV in Google Drive first.")


async def _download_csv(file_id: str, mime_type: str) -> str:
    if mime_type == "application/vnd.google-apps.spreadsheet":
        arguments: dict[str, Any] = {"file_id": file_id, "export_mime_type": "text/csv"}
    elif mime_type.startswith("text/") or "csv" in mime_type:
        arguments = {"file_id": file_id}
    else:
        raise agents.ToolError(
            f"Only CSV files are supported: this file has MIME type '{mime_type}'. In Google Drive, use "
            "File -> Download -> Comma-separated values (.csv), upload the CSV, then share its link."
        )
    payload = await _call_drive_tool(DOWNLOAD_TOOL, arguments)
    if payload is None or not payload.get("content"):
        raise agents.ToolError(
            "Google Drive returned no file content. The file may be empty or larger than the "
            "connector's download limit."
        )
    try:
        data = base64.b64decode(payload["content"])
    except Exception as exc:
        raise agents.ToolError("Could not decode the downloaded file.") from exc
    if len(data) > MAX_CSV_BYTES:
        raise agents.ToolError(
            f"The file is {len(data) / 1e6:.1f} MB, above the {MAX_CSV_BYTES // 1_000_000} MB limit."
        )
    return _decode_text(data)


_DATASETS: dict[str, "_Dataset"] = {}
_CURRENT: str | None = None


class _Dataset:
    def __init__(self, file_id: str, title: str, mime_type: str, frame: pd.DataFrame) -> None:
        self.file_id = file_id
        self.title = title
        self.mime_type = mime_type
        self.df = frame


def _sniff_parse(text: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for sep in (",", ";", "\t"):
        try:
            frame = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            if frame.shape[1] > 1:
                return frame
            last_error = last_error or ValueError("parsed as a single column")
        except Exception as exc:
            last_error = exc
    raise agents.ToolError(f"Could not parse the file as CSV ({last_error}).")


def _smart_coerce(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        series = out[column]
        if series.dtype == object or str(series.dtype) == "string":
            cleaned = series.astype("string").str.strip().str.replace(",", "", regex=False)
            non_null = cleaned.notna() & (cleaned != "")
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if non_null.any() and numeric[non_null].notna().sum() / non_null.sum() >= 0.8:
                out[column] = numeric
                continue
            dates = pd.to_datetime(series, errors="coerce", format="mixed")
            if non_null.any() and dates[non_null].notna().sum() / non_null.sum() >= 0.8:
                out[column] = dates
    return out


def _remember(dataset: _Dataset, raw_text: str) -> None:
    _DATASETS[dataset.file_id] = dataset
    global _CURRENT
    _CURRENT = dataset.file_id
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(dataset.file_id).write_text(raw_text, encoding="utf-8")
    except OSError:
        pass


def _resolve_dataset(file_id: str | None) -> _Dataset:
    if file_id:
        dataset = _DATASETS.get(file_id)
        if dataset:
            global _CURRENT
            _CURRENT = file_id
            return dataset
        cache = _cache_path(file_id)
        if cache.exists():
            frame = _smart_coerce(_sniff_parse(cache.read_text(encoding="utf-8")))
            restored = _Dataset(file_id, cache.stem, "text/csv", frame)
            _DATASETS[file_id] = restored
            _CURRENT = file_id
            return restored
        raise agents.ToolError(
            "That file is not loaded in this worker's memory. Call load_dataset with its Drive link again."
        )
    if _CURRENT and _CURRENT in _DATASETS:
        return _DATASETS[_CURRENT]
    if _DATASETS:
        raise agents.ToolError(
            "No dataset is currently selected. Pass file_id, or call load_dataset again."
        )
    raise agents.ToolError(
        "No dataset is loaded yet. Ask the user for a Google Drive link to a CSV file, "
        "then call load_dataset with it."
    )


def _find_column(frame: pd.DataFrame, name: str) -> str:
    if name in frame.columns:
        return str(name)
    normalized = re.sub(r"[\s_]+", "", str(name)).lower()
    for column in frame.columns:
        if re.sub(r"[\s_]+", "", str(column)).lower() == normalized:
            return str(column)
    options = ", ".join(repr(str(c)) for c in frame.columns)
    raise agents.ToolError(f"No column named {name!r}. Available columns: {options}.")


def _numeric_like(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series
    coerced = pd.to_numeric(series.astype("string").str.replace(",", "", regex=False), errors="coerce")
    non_null = series.notna()
    if non_null.any() and coerced[non_null].notna().sum() / non_null.sum() >= 0.8:
        return coerced
    return None


def _datetime_like(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    non_null = series.notna()
    if non_null.any() and parsed[non_null].notna().sum() / non_null.sum() >= 0.8:
        return parsed
    return None


def _ordered_mask(series: pd.Series, op: str, value: str | float | int) -> pd.Series:
    numeric = _numeric_like(series)
    if numeric is not None:
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            raise agents.ToolError(f"Value {value!r} is not a number for numeric column {series.name!r}.")
        if op == "greater_than":
            return numeric > threshold
        if op == "greater_or_equal":
            return numeric >= threshold
        if op == "less_than":
            return numeric < threshold
        return numeric <= threshold
    datetime = _datetime_like(series)
    if datetime is not None:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed) and value is not None:
            raise agents.ToolError(f"Could not parse {value!r} as a date/time.")
        if op == "greater_than":
            return datetime > parsed
        if op == "greater_or_equal":
            return datetime >= parsed
        if op == "less_than":
            return datetime < parsed
        return datetime <= parsed
    raise agents.ToolError(
        f"Column {series.name!r} holds text; use equals, contains, starts_with or ends_with to filter it."
    )


def _match_mask(series: pd.Series, value: str | float | int) -> pd.Series:
    numeric = _numeric_like(series)
    if numeric is not None:
        try:
            return numeric == float(value)
        except (TypeError, ValueError):
            raise agents.ToolError(f"Cannot compare numeric column {series.name!r} with {value!r}.")
    datetime = _datetime_like(series)
    if datetime is not None:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise agents.ToolError(f"Could not parse {value!r} as a date/time for column {series.name!r}.")
        return datetime == parsed
    return series.astype("string").str.strip().str.casefold() == str(value).strip().casefold()


def _rows_payload(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    return json.loads(frame.head(limit).to_json(orient="records", date_format="iso", double_precision=6))


@agents.tool(model_access="direct")
async def load_dataset(
    drive_link: Annotated[str, Field(description="Google Drive link or file id of the CSV file.")],
) -> dict[str, Any]:
    """Load a CSV file from the user's Google Drive and select it as the dataset for follow-up questions.

    Accepts any Google Drive link (drive.google.com or docs.google.com URL) or a raw file id.
    Google Sheets are exported to CSV automatically. Returns the row count and every column with its type.
    """
    file_id = _extract_file_id(drive_link)
    metadata = await _call_drive_tool(METADATA_TOOL, {"file_id": file_id})
    if not metadata or not metadata.get("id"):
        raise agents.ToolError(
            "Google Drive has no file with that id (metadata lookup failed). Check the link and that the "
            "file is accessible to the account connected to the Google Drive connector."
        )
    title = str(metadata.get("title") or "untitled")
    mime_type = str(metadata.get("mime_type") or "text/csv")
    text = await _download_csv(file_id, mime_type)
    frame = _smart_coerce(_sniff_parse(text))
    dataset = _Dataset(file_id, title, mime_type, frame)
    _remember(dataset, text)
    null_counts = frame.isna().sum()
    return {
        "file_id": file_id,
        "title": title,
        "mime_type": mime_type,
        "rows": int(len(frame)),
        "columns": [
            {"name": str(c), "dtype": str(frame[c].dtype), "nulls": int(null_counts[c])} for c in frame.columns
        ],
    }


@agents.tool(model_access="direct")
async def describe_columns(
    file_id: Annotated[str | None, Field(description="Optional file id of a loaded dataset; defaults to the last loaded file.")] = None,
) -> list[dict[str, Any]]:
    """Describe the columns of the loaded dataset: type, nulls, uniqueness, and per-column statistics.

    Numeric columns get min/mean/median/max; date columns get min/max; text columns get their
    most frequent values. Use this first to learn what the data contains.
    """
    dataset = _resolve_dataset(file_id)
    frame = dataset.df
    descriptions: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        entry: dict[str, Any] = {
            "column": str(column),
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "nulls": int(series.isna().sum()),
            "unique_values": int(series.nunique()),
        }
        numeric = _numeric_like(series)
        datetime = _datetime_like(series)
        if numeric is not None:
            entry["kind"] = "numeric"
            entry["min"] = round(float(numeric.min()), 6)
            entry["mean"] = round(float(numeric.mean()), 6)
            entry["median"] = round(float(numeric.median()), 6)
            entry["max"] = round(float(numeric.max()), 6)
            entry["std"] = round(float(numeric.std()), 6) if numeric.count() > 1 else None
        elif datetime is not None:
            entry["kind"] = "date"
            entry["min"] = str(datetime.min())
            entry["max"] = str(datetime.max())
        else:
            entry["kind"] = "text"
            top = series.astype("string").value_counts().head(5)
            entry["top_values"] = [{"value": None if pd.isna(k) else str(k), "count": int(v)} for k, v in top.items()]
        if entry["unique_values"] <= 10:
            entry["distinct_values"] = [str(v) for v in series.dropna().unique()[:10]]
        descriptions.append(entry)
    return descriptions


@agents.tool(model_access="direct")
async def filter_rows(
    column: Annotated[str, Field(description="Column name to filter on, exactly as returned by describe_columns.")],
    operation: Annotated[
        Literal[
            "equals",
            "not_equals",
            "contains",
            "not_contains",
            "starts_with",
            "ends_with",
            "in",
            "greater_than",
            "greater_or_equal",
            "less_than",
            "less_or_equal",
            "between",
            "is_null",
            "not_null",
        ],
        Field(description="Filter operation."),
    ],
    value: Annotated[str | float | int | None, Field(description="Scalar value compared against; required unless the operation is in, is_null or not_null.")] = None,
    value_2: Annotated[str | float | int | None, Field(description="Second value; only for between (inclusive range value..value_2).")] = None,
    value_list: Annotated[list[str | float] | None, Field(description="List of values; only for the in operation.")] = None,
    file_id: Annotated[str | None, Field(description="Optional file id when several files are loaded.")] = None,
    limit: Annotated[int, Field(ge=1, le=100, description="Maximum number of matching rows to return, 1-100. Default 20.")] = 20,
) -> dict[str, Any]:
    """Filter rows of the loaded dataset with one fixed condition; returns the total match count and matching rows.

    Fixed operations only, no code runs. To combine several conditions, call it several times or
    use groupby_aggregate.
    """
    dataset = _resolve_dataset(file_id)
    frame = dataset.df
    series = frame[_find_column(frame, column)]
    limit = max(1, min(int(limit), 100))
    if operation == "is_null":
        mask = series.isna()
    elif operation == "not_null":
        mask = series.notna()
    elif operation == "equals":
        if value is None:
            raise agents.ToolError("equals needs a value.")
        mask = _match_mask(series, value)
    elif operation == "not_equals":
        if value is None:
            raise agents.ToolError("not_equals needs a value.")
        mask = ~_match_mask(series, value)
    elif operation == "contains":
        mask = series.astype("string").str.contains(str(value), case=False, na=False, regex=False)
    elif operation == "not_contains":
        mask = ~series.astype("string").str.contains(str(value), case=False, na=False, regex=False)
    elif operation == "starts_with":
        mask = series.astype("string").str.strip().str.startswith(str(value), na=False)
    elif operation == "ends_with":
        mask = series.astype("string").str.strip().str.endswith(str(value), na=False)
    elif operation == "in":
        values = value_list if value_list is not None else ([value] if value is not None else [])
        if not values:
            raise agents.ToolError("in needs value_list (or a single value).")
        mask = series.astype("string").str.strip().str.casefold().isin([str(v).strip().casefold() for v in values])
    elif operation == "between":
        if value is None or value_2 is None:
            raise agents.ToolError("between needs both value and value_2.")
        mask = _ordered_mask(series, "greater_or_equal", value) & _ordered_mask(series, "less_or_equal", value_2)
    elif operation in ("greater_than", "greater_or_equal", "less_than", "less_or_equal"):
        if value is None:
            raise agents.ToolError(f"{operation} needs a value.")
        mask = _ordered_mask(series, operation, value)
    else:
        raise agents.ToolError(f"Unknown operation {operation!r}.")
    matching = frame[mask]
    operation_label = f"{column} {operation}" + (f" {value}" if value is not None else "") + (f" .. {value_2}" if operation == "between" else "")
    return {
        "file_id": dataset.file_id,
        "title": dataset.title,
        "operation": operation_label,
        "total_rows": int(len(frame)),
        "total_matches": int(mask.sum()),
        "returned_rows": int(min(len(matching), limit)),
        "columns": [str(c) for c in frame.columns],
        "rows": _rows_payload(matching, limit),
    }


def _numeric_for_agg(series: pd.Series, column: str, func: str) -> pd.Series:
    if func in ("sum", "mean", "median", "std", "min", "max") and not pd.api.types.is_numeric_dtype(series):
        coerced = pd.to_numeric(series.astype("string").str.replace(",", "", regex=False), errors="coerce")
        if coerced.notna().sum() == 0:
            raise agents.ToolError(
                f"Cannot compute {func} on column {column!r}: it has no numeric values. Use count or nunique."
            )
        return coerced
    return series


@agents.tool(model_access="direct")
async def groupby_aggregate(
    group_by: Annotated[str | list[str], Field(description="One column name or a list of column names to group by.")],
    aggregations: Annotated[
        list[dict[str, str]],
        Field(description="Aggregations to compute, e.g. [{'column': 'revenue', 'func': 'sum'}]. func is one of: count, nunique, sum, mean, median, min, max, std."),
    ],
    order_by: Annotated[str | None, Field(description="Optional result column to sort by, e.g. 'sum_revenue'.")] = None,
    descending: Annotated[bool, Field(description="Sort direction when order_by is given. Default: desc=True.")] = True,
    limit: Annotated[int, Field(ge=1, le=200, description="Maximum number of groups to return, 1-200. Default 50.")] = 50,
    file_id: Annotated[str | None, Field(description="Optional file id when several files are loaded.")] = None,
) -> dict[str, Any]:
    """Group the loaded dataset by one or more columns and compute fixed aggregates (count, sum, mean, ...).

    Returns one row per group with the requested aggregates. Result columns are named '<func>_<column>'
    (e.g. 'sum_revenue', count of rows is 'count_rows'); use those names in order_by.
    """
    dataset = _resolve_dataset(file_id)
    frame = dataset.df
    if isinstance(group_by, str):
        group_columns = [_find_column(frame, group_by)]
    else:
        if not group_by:
            raise agents.ToolError("group_by must name at least one column.")
        group_columns = [_find_column(frame, name) for name in group_by]
    if not aggregations:
        aggregations = [{"column": frame.columns[0], "func": "count"}]
    limit = max(1, min(int(limit), 200))

    work = frame[group_columns].copy()
    agg_map: dict[str, str] = {}
    used_names: set[str] = set()
    for entry in aggregations:
        raw_column = str(entry.get("column", ""))
        column = _find_column(frame, raw_column)
        func = str(entry.get("func", "count")).lower()
        if func not in AGGREGATION_FUNCS:
            raise agents.ToolError(f"Unknown aggregation func {func!r}. Allowed: {', '.join(AGGREGATION_FUNCS)}.")
        if func == "count":
            name = "count_rows"
            if name not in used_names:
                work[name] = 1
                used_names.add(name)
            agg_map[name] = "count"
        else:
            name = f"{func}_{column}"
            while name in used_names:
                name += "_"
            used_names.add(name)
            work[name] = _numeric_for_agg(frame[column], column, func)
            agg_map[name] = "nunique" if func == "nunique" else func

    grouped = work.groupby(group_columns, dropna=False, observed=True).agg(agg_map).reset_index()
    total_groups = int(len(grouped))
    if order_by:
        target = str(order_by)
        if target not in grouped.columns:
            raise agents.ToolError(
                f"order_by {target!r} is not a result column. Result columns: "
                + ", ".join(str(c) for c in grouped.columns)
            )
        grouped = grouped.sort_values(by=target, ascending=not descending, na_position="last")
    records = json.loads(grouped.head(limit).to_json(orient="records", date_format="iso", double_precision=6))
    return {
        "file_id": dataset.file_id,
        "title": dataset.title,
        "group_by": group_columns,
        "aggregations": [f"{name}={func}" for name, func in agg_map.items()],
        "total_rows": int(len(frame)),
        "total_groups": total_groups,
        "returned_groups": len(records),
        "columns": [str(c) for c in grouped.columns],
        "rows": records,
    }


agent = agents.Agent(
    model="mistral-large-4",
    name="rag1pilot",
    instructions=(
        "You are Rag1Pilot, a conversational data analyst for CSV files in the user's Google Drive.\n"
        "The user gives you a Drive link; you load it and answer follow-up questions in the same conversation.\n\n"
        "How you work:\n"
        "- You have exactly four tools: load_dataset, describe_columns, filter_rows, groupby_aggregate. "
        "Every number you state must come from one of these tools. Never invent, extrapolate or estimate numbers.\n"
        "- When a file is first mentioned, call load_dataset with the link or file id, then describe_columns "
        "to understand the schema before answering.\n"
        "- Remember the loaded file and everything discussed; follow-ups do not need the link again. The "
        "loaded dataset stays selected. Pass file_id only when several datasets are loaded.\n"
        "- Match questions to tools: 'how many rows match X' -> filter_rows (use total_matches); "
        "'average of X by Y' -> groupby_aggregate with func mean; 'total X per Y' -> sum; "
        "'how many X per Y' -> count; 'what's in the data' -> describe_columns.\n"
        "- Never use run_typescript to compute anything: no arithmetic, no re-implementing analysis, "
        "no data manipulation. Use it at most to chain the four analysis tools in one step; all "
        "numbers must come from tool results.\n"
        "- If a tool returns an error, read the message and recover: fix the column name against the "
        "describe_columns output, adjust the operation, or explain precisely what is not possible.\n"
        "- If no dataset is loaded and the user named a file without a link, use the Google Drive "
        "connector's search_files tool to find it, then load_dataset with the found file id or its "
        "view_url. Apart from that, use Google Drive tools only for metadata lookups: never call "
        "create_file, copy_file or update_file, and never dump file content into the conversation, "
        "because load_dataset gives the analysis tools the data.\n"
        "- If no dataset is loaded and no file can be found, ask the user for a Google Drive link to a "
        "CSV file.\n"
        "- Only CSV files are supported; Google Sheets are fine (loaded as CSV). For other formats, ask "
        "the user to export CSV.\n"
        "- Present results as short readable tables or bullets with the real numbers, rounded sensibly "
        "(e.g. 1,234.5). When useful, relate a filtered or grouped count back to the total row count.\n"
        "- End each answer with one short suggestion of a useful follow-up. Never suggest capabilities "
        "you do not have: no plotting, no arbitrary computations, no other file formats."
    ),
    tools=[load_dataset, describe_columns, filter_rows, groupby_aggregate],
    connectors=[agents.connector("google_drive_mcp")],
    builtins=agents.Builtins(
        skills=False,
        subagents=False,
        programmatic_tool_calling=agents.ProgrammaticToolCallingConfig(sleep=False),
    ),
)
