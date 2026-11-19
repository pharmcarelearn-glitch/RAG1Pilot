"""Conversational data analyst over CSV files in Google Drive.

Fixed tools only: load a CSV, describe its columns, filter rows and compute
group-by aggregates. There is no arbitrary code execution on the data: every
analysis goes through one of the typed tools below.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import re
import threading
import time
from collections import OrderedDict
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from mistralai import agents

MAX_CSV_BYTES = 20 * 1024 * 1024
MAX_DATASETS = 10
MAX_FILTER_ROWS = 100
MAX_GROUP_ROWS = 200
SAMPLE_CELL_CHARS = 80


class FilterCondition(BaseModel):
    column: str
    operator: Literal[
        "eq",
        "ne",
        "gt",
        "ge",
        "lt",
        "le",
        "contains",
        "startswith",
        "endswith",
        "in",
        "not_in",
        "is_null",
        "not_null",
    ]
    value: (
        str | int | float | bool | list[str | int | float] | None
    ) = None
    case_sensitive: bool = False


class SortSpec(BaseModel):
    column: str
    descending: bool = False


class AggregateSpec(BaseModel):
    column: str | None = Field(
        default=None,
        description="Column to aggregate. Leave None only for function='count'.",
    )
    function: Literal[
        "count",
        "nunique",
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "var",
        "first",
        "last",
    ]
    result_name: str | None = None


class _Dataset:
    def __init__(self, name: str, file_id: str, df: pd.DataFrame, csv_bytes: bytes):
        self.name = name
        self.file_id = file_id
        self.df = df
        self.csv_bytes = csv_bytes
        self.loaded_at = time.time()


_registry: OrderedDict[str, _Dataset] = OrderedDict()
_lock = threading.Lock()


def _put_dataset(ds: _Dataset) -> None:
    with _lock:
        _registry.pop(ds.name, None)
        _registry[ds.name] = ds
        while len(_registry) > MAX_DATASETS:
            _registry.popitem(last=False)


def _get_dataset(dataset: str) -> _Dataset:
    with _lock:
        ds = _registry.get(dataset)
        if ds is not None:
            _registry.move_to_end(dataset)
        return ds


def _require_dataset(dataset: str) -> pd.DataFrame:
    ds = _get_dataset(dataset)
    if ds is None:
        names = _registry_names()
        hint = f"Loaded datasets: {names}." if names else "No dataset is loaded yet."
        raise agents.ToolError(
            f"Unknown dataset '{dataset}'. {hint} "
            "Load a file first with load_csv, or list datasets with list_loaded_datasets."
        )
    return ds.df


def _registry_names() -> list[str]:
    with _lock:
        return list(_registry.keys())


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise agents.ToolError(
            f"Column '{column}' does not exist. Available columns: {list(df.columns)}."
        )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return None
        return round(number, 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    return str(value)


def _records(df: pd.DataFrame, limit: int) -> tuple[dict[str, Any], int]:
    sample = df.head(limit)
    rows = []
    for record in sample.to_dict(orient="records"):
        rows.append({col: _json_safe(val) for col, val in record.items()})
    return rows, len(df)


def _apply_filters(df: pd.DataFrame, filters: list[FilterCondition]) -> pd.DataFrame:
    for cond in filters:
        _require_column(df, cond.column)
        series = df[cond.column]
        op = cond.operator
        if op == "is_null":
            mask = series.isna()
        elif op == "not_null":
            mask = series.notna()
        elif op in ("in", "not_in"):
            if not isinstance(cond.value, list):
                raise agents.ToolError(
                    f"Operator '{op}' requires a list value."
                )
            values = [_json_safe(v) for v in cond.value]
            if pd.api.types.is_numeric_dtype(series):
                numeric = []
                for v in values:
                    try:
                        numeric.append(float(v))  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        continue
                mask = series.isin(numeric)
            elif cond.case_sensitive:
                mask = series.isin(values)
            else:
                lowered = {str(v).strip().lower() for v in values}
                mask = series.astype(str).str.strip().str.lower().isin(lowered)
            if op == "not_in":
                mask = ~mask
        else:
            value = cond.value
            if pd.api.types.is_numeric_dtype(series) and isinstance(value, str):
                try:
                    value = float(value)
                except ValueError as exc:
                    raise agents.ToolError(
                        f"Value '{cond.value}' is not numeric for column "
                        f"'{cond.column}' ({series.dtype})."
                    ) from exc
            if op in ("contains", "startswith", "endswith"):
                text = series.astype(str)
                needle = str(value)
                if not cond.case_sensitive:
                    text = text.str.lower()
                    needle = needle.lower()
                if op == "contains":
                    mask = text.str.contains(needle, regex=False)
                elif op == "startswith":
                    mask = text.str.startswith(needle)
                else:
                    mask = text.str.endswith(needle)
            elif op == "eq":
                mask = _equals(series, value, cond.case_sensitive)
            elif op == "ne":
                mask = ~_equals(series, value, cond.case_sensitive)
            else:
                if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                    try:
                        value = float(value) if isinstance(value, str) else value  # type: ignore[assignment]
                    except (TypeError, ValueError):
                        pass
                try:
                    if op == "gt":
                        mask = series > value
                    elif op == "ge":
                        mask = series >= value
                    elif op == "lt":
                        mask = series < value
                    elif op == "le":
                        mask = series <= value
                    else:
                        raise agents.ToolError(f"Unknown operator '{op}'.")
                except TypeError:
                    raise agents.ToolError(
                        f"Cannot compare column '{cond.column}' "
                        f"(dtype {series.dtype}) with value {cond.value!r} "
                        f"using '{op}'."
                    )
        df = df[mask.fillna(False)]
    return df


def _equals(series: pd.Series, value: Any, case_sensitive: bool) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        try:
            return series == float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return pd.Series(False, index=series.index)
    if isinstance(value, (bool, np.bool_)):
        return series == value
    if not case_sensitive:
        return (
            series.astype(str).str.strip().str.lower()
            == str(value).strip().lower()
        )
    return series == value


@agents.tool
def load_csv(
    dataset: Annotated[str, Field(description="Short name for the dataset, used by every analysis tool afterwards. Derive it from the file title, e.g. 'patients_profiles'.")],
    file_id: Annotated[str, Field(description="Google Drive file id of the CSV.")],
    csv_base64: Annotated[str, Field(description="Base64-encoded CSV bytes, exactly as returned by google_drive_mcp_download_file_content (field 'content'). Do not decode, truncate or modify it.")],
) -> dict:
    """Register a CSV from Google Drive as the working dataset. Returns a summary only; the file content itself must never be echoed back."""
    if not csv_base64:
        raise agents.ToolError("csv_base64 is empty: pass the 'content' field returned by google_drive_mcp_download_file_content.")
    try:
        csv_bytes = base64.b64decode(csv_base64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise agents.ToolError(f"csv_base64 is not valid base64: {exc}") from exc
    if not csv_bytes:
        raise agents.ToolError("Decoded CSV is empty.")
    if len(csv_bytes) > MAX_CSV_BYTES:
        raise agents.ToolError(
            f"CSV is {len(csv_bytes)} bytes; the limit is {MAX_CSV_BYTES}."
        )
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig", sep=None, engine="python")
    except Exception as exc:
        raise agents.ToolError(f"Could not parse the file as CSV: {exc}") from exc
    if df.empty:
        raise agents.ToolError("The CSV parsed to 0 rows.")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset).strip("_")[:64] or "dataset"
    ds = _Dataset(safe_name, file_id, df, csv_bytes)
    _put_dataset(ds)
    rows, _ = _records(df.head(5), 5)
    return {
        "dataset": safe_name,
        "file_id": file_id,
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing_per_column": {str(c): int(n) for c, n in df.isna().sum().items()},
        "sample_rows": rows,
    }


@agents.tool
def list_loaded_datasets() -> dict:
    """List the datasets currently loaded in this session."""
    names = _registry_names()
    return {"datasets": names}


@agents.tool
def describe_columns(dataset: Annotated[str, Field(description="Dataset name, as returned by load_csv.")]) -> dict:
    """Describe every column of a loaded dataset: dtype, missing count, distinct count, and for numeric columns min/max/mean or for text columns the most frequent values."""
    df = _require_dataset(dataset)
    columns: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        info: dict[str, Any] = {
            "column": str(col),
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "distinct": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            info["min"] = _json_safe(series.min())
            info["max"] = _json_safe(series.max())
            info["mean"] = _json_safe(series.mean())
            info["median"] = _json_safe(series.median())
        else:
            top = series.astype(str).str.slice(0, SAMPLE_CELL_CHARS).value_counts(dropna=True).head(5)
            info["top_values"] = [
                {"value": str(v), "count": int(c)} for v, c in top.items()
            ]
        columns.append(info)
    return {"dataset": dataset, "rows": int(len(df)), "columns": columns}


@agents.tool
def filter_rows(
    dataset: Annotated[str, Field(description="Dataset name, as returned by load_csv.")],
    filters: Annotated[list[FilterCondition], Field(description="Conditions combined with AND. Empty list returns an unfiltered preview.")],
    columns: Annotated[list[str] | None, Field(description="Optional subset of columns to return.")] = None,
    sort_by: Annotated[list[SortSpec] | None, Field(description="Optional sort applied before row-limiting.")] = None,
    limit: Annotated[int, Field(description="Maximum rows to return, between 1 and 100.")] = 20,
) -> dict:
    """Filter rows of a loaded dataset and return the matching rows plus their total count. Use it to inspect rows, not to count groups."""
    df = _require_dataset(dataset)
    if columns:
        for col in columns:
            _require_column(df, col)
        df = df[columns]
    df = _apply_filters(df, filters)
    if sort_by:
        for spec in sort_by:
            _require_column(df, spec.column)
        df = df.sort_values(
            by=[spec.column for spec in sort_by],
            ascending=[not spec.descending for spec in sort_by],
        )
    rows, total = _records(df, max(1, min(int(limit), MAX_FILTER_ROWS)))
    return {
        "dataset": dataset,
        "total_matching": total,
        "returned": len(rows),
        "rows": rows,
    }


@agents.tool
def group_by_aggregate(
    dataset: Annotated[str, Field(description="Dataset name, as returned by load_csv.")],
    group_by: Annotated[list[str], Field(description="Columns to group by, one or more.")],
    aggregations: Annotated[list[AggregateSpec], Field(description="Aggregations to compute per group. For a plain row count use function='count' with column left empty.")],
    filters: Annotated[list[FilterCondition], Field(description="Optional conditions applied before grouping, combined with AND.")] = [],
    limit: Annotated[int, Field(description="Maximum groups to return, between 1 and 200. Groups with the most rows are kept first.")] = 50,
) -> dict:
    """Group a loaded dataset by one or more columns and compute aggregates per group (count, nunique, sum, mean, median, min, max, std, var, first, last). Use this for questions like 'average revenue per region' or 'how many rows per domain'."""
    df = _require_dataset(dataset)
    if not group_by:
        raise agents.ToolError("group_by must list at least one column.")
    for col in group_by:
        _require_column(df, col)
    if not aggregations:
        raise agents.ToolError("aggregations must list at least one aggregation.")
    df = _apply_filters(df, filters)
    named: dict[str, pd.NamedAgg] = {}
    count_result_name: str | None = None
    for spec in aggregations:
        if spec.column is None:
            if spec.function != "count":
                raise agents.ToolError(
                    "Aggregations without a column only support function='count'."
                )
            count_result_name = spec.result_name or "count"
            continue
        _require_column(df, spec.column)
        result_name = spec.result_name or f"{spec.function}_{spec.column}"
        named[result_name] = pd.NamedAgg(column=spec.column, aggfunc=spec.function)
    try:
        grouped = df.groupby(group_by, dropna=False).agg(**named).reset_index()
        if count_result_name is not None:
            sizes = df.groupby(group_by, dropna=False).size().reset_index(name=count_result_name)
            grouped = grouped.merge(sizes, on=group_by, how="left")
    except Exception as exc:
        raise agents.ToolError(f"Aggregation failed: {exc}") from exc
    total_groups = int(len(grouped))
    limit_value = max(1, min(int(limit), MAX_GROUP_ROWS))
    if count_result_name is not None and limit_value < total_groups:
        grouped = grouped.sort_values(by=count_result_name, ascending=False)
    limit_value = max(1, min(int(limit), MAX_GROUP_ROWS))
    rows, _ = _records(grouped, limit_value)
    return {
        "dataset": dataset,
        "total_groups": total_groups,
        "returned": len(rows),
        "groups": rows,
    }


agent = agents.Agent(
    model="mistral-large-4",
    name="rag1pilot",
    instructions=(
        "You are a conversational data analyst. The user asks questions in plain "
        "language about CSV files in their Google Drive, and you answer with real "
        "numbers from the data.\n\n"
        "The user's conversation memory is yours: once a file is loaded you keep "
        "referring to it by its dataset name in follow-up questions. Never ask the "
        "user to re-explain or re-share something already discussed in the session.\n\n"
        "Loading a file, when the user gives a Drive link, a file title or asks to "
        "analyze a file:\n"
        "1. Get the file id: extract it from the link (the long id in "
        "/file/d/<FILE_ID>/ or ?id=<FILE_ID>), or find it with the connector tool "
        "google_drive_mcp_search_files (query the title, add mimeType = 'text/csv' "
        "when the user asks for a CSV). Confirm the file matches what the user means "
        "when several files are plausible.\n"
        "2. Write one run_typescript program that calls "
        "google_drive_mcp_download_file_content({file_id}) and then "
        "tools.client.load_csv({dataset: <short name from the file title>, file_id: "
        "<the file id>, csv_base64: <the 'content' field of the download result>}). "
        "Do not modify or truncate the 'content' field. The program's final result "
        "must be the object returned by load_csv, nothing else.\n"
        "3. NEVER put file content, base64 strings, or CSV rows in your own reply. "
        "The only data in your answers comes from tool results of load_csv, "
        "describe_columns, filter_rows and group_by_aggregate.\n"
        "4. After loading, briefly state the row count, the columns, and one thing "
        "worth noticing, then answer the user's first question if they asked one.\n\n"
        "Answering questions:\n"
        "- Run describe_columns before the first question on a dataset so you know "
        "the types; prefer it over guessing column names.\n"
        "- Use group_by_aggregate for 'per X', 'by X', 'average/sum/count per "
        "group' questions; use filter_rows for row-level lookups; use "
        "list_loaded_datasets when unsure which datasets exist. Chain several calls "
        "when a question needs more than one step, and compute derived numbers "
        "only with standard arithmetic on returned aggregates; say so when you do.\n"
        "- Quote the exact numbers the tools returned. Never invent, extrapolate or "
        "round beyond what the tool output supports; give the underlying counts "
        "(rows, groups, missing values) alongside any average or share.\n"
        "- If the data cannot answer a question, say precisely what is missing "
        "instead of guessing.\n"
        "- Mention the dataset name you are talking about so that follow-ups stay "
        "unambiguous; use the most recently loaded dataset only while it is the "
        "clear topic of the conversation.\n\n"
        "You have no code execution on the data beyond the fixed tools: never claim "
        "to have run computations other than those tools and simple arithmetic on "
        "their results."
    ),
    tools=[load_csv, list_loaded_datasets, describe_columns, filter_rows, group_by_aggregate],
    connectors=[agents.connector("google_drive_mcp")],
)


if __name__ == "__main__":
    pass
