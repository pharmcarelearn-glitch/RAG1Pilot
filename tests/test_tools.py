import base64
import io

import pandas as pd
import pytest
from mistralai import agents

from main import AggregateSpec, FilterCondition, SortSpec, describe_columns, filter_rows, group_by_aggregate, list_loaded_datasets, load_csv


@pytest.fixture()
def csv_b64() -> str:
    df = pd.DataFrame(
        {
            "region": ["north", "south", "north", "east", "south", "North"],
            "product": ["alpha", "alpha", "beta", "beta", "gamma", "gamma"],
            "units": [10, 5, 0, 7, None, 3],
            "revenue": [100.5, 50.25, 0.0, 70.0, 30.0, 33.5],
        }
    )
    return base64.b64encode(df.to_csv(index=False).encode()).decode()


def _load(name: str, csv_b64: str):
    return load_csv(dataset=name, file_id="test-file-id", csv_base64=csv_b64)


def test_load_returns_summary(csv_b64):
    summary = _load("sales", csv_b64)
    assert summary["dataset"] == "sales"
    assert summary["rows"] == 6
    assert summary["columns"] == ["region", "product", "units", "revenue"]
    assert summary["missing_per_column"]["units"] == 1
    assert len(summary["sample_rows"]) == 5


def test_load_strips_utf8_bom(csv_b64):
    with_bom = base64.b64encode(b"\xef\xbb\xbf" + base64.b64decode(csv_b64)).decode()
    summary = load_csv(dataset="bom", file_id="x", csv_base64=with_bom)
    assert summary["columns"][0] == "region"


def test_list_loaded_datasets(csv_b64):
    _load("sales", csv_b64)
    assert "sales" in list_loaded_datasets()["datasets"]


def test_describe_columns(csv_b64):
    _load("sales", csv_b64)
    result = describe_columns(dataset="sales")
    by_name = {c["column"]: c for c in result["columns"]}
    assert by_name["revenue"]["min"] == 0.0
    assert by_name["revenue"]["mean"] == pytest.approx(47.375)
    assert by_name["region"]["top_values"][0]["value"] in ("north", "North")
    assert by_name["units"]["missing"] == 1


def test_filter_numeric_comparison(csv_b64):
    _load("sales", csv_b64)
    result = filter_rows(
        dataset="sales",
        filters=[FilterCondition(column="revenue", operator="gt", value=50)],
    )
    assert result["total_matching"] == 3
    assert all(row["revenue"] > 50 for row in result["rows"])


def test_filter_contains_case_insensitive(csv_b64):
    _load("sales", csv_b64)
    result = filter_rows(
        dataset="sales",
        filters=[FilterCondition(column="region", operator="contains", value="NOR")],
    )
    assert result["total_matching"] == 3


def test_filter_in_operator(csv_b64):
    _load("sales", csv_b64)
    result = filter_rows(
        dataset="sales",
        filters=[FilterCondition(column="product", operator="in", value=["alpha", "gamma"])],
    )
    assert result["total_matching"] == 4


def test_filter_sort_and_limit(csv_b64):
    _load("sales", csv_b64)
    result = filter_rows(
        dataset="sales",
        filters=[],
        sort_by=[SortSpec(column="revenue", descending=True)],
        limit=2,
    )
    assert result["total_matching"] == 6
    assert result["returned"] == 2
    assert result["rows"][0]["revenue"] == 100.5


def test_filter_unknown_column_raises(csv_b64):
    _load("sales", csv_b64)
    with pytest.raises(agents.ToolError):
        filter_rows(dataset="sales", filters=[FilterCondition(column="nope", operator="eq", value=1)])


def test_group_by_aggregate(csv_b64):
    _load("sales", csv_b64)
    result = group_by_aggregate(
        dataset="sales",
        group_by=["region"],
        aggregations=[
            AggregateSpec(function="count"),
            AggregateSpec(column="revenue", function="mean"),
        ],
    )
    groups = {g["region"].lower(): g for g in result["groups"]}
    assert groups["north"]["count"] == 2
    assert groups["north"]["mean_revenue"] == pytest.approx(50.25)
    assert groups["east"]["count"] == 1
    assert groups["east"]["mean_revenue"] == pytest.approx(70.0)


def test_group_by_with_filters(csv_b64):
    _load("sales", csv_b64)
    result = group_by_aggregate(
        dataset="sales",
        group_by=["product"],
        aggregations=[AggregateSpec(column="revenue", function="sum")],
        filters=[FilterCondition(column="revenue", operator="ge", value=40)],
    )
    sums = {g["product"]: g["sum_revenue"] for g in result["groups"]}
    assert sums["alpha"] == pytest.approx(150.75)
    assert sums["beta"] == pytest.approx(70.0)
    assert "gamma" not in sums


def test_group_by_requires_group_by(csv_b64):
    _load("sales", csv_b64)
    with pytest.raises(agents.ToolError):
        group_by_aggregate(dataset="sales", group_by=[], aggregations=[AggregateSpec(function="count")])


def test_unknown_dataset_raises():
    with pytest.raises(agents.ToolError, match="Unknown dataset"):
        describe_columns(dataset="does_not_exist")


def test_invalid_base64_raises():
    with pytest.raises(agents.ToolError, match="not valid base64"):
        load_csv(dataset="bad", file_id="x", csv_base64="!!!not-base64!!!")


def test_load_sound_data_roundtrip():
    buffer = io.StringIO("a,b\n1,x\n2,y\n")
    summary = load_csv(
        dataset="sound", file_id="x", csv_base64=base64.b64encode(buffer.getvalue().encode()).decode()
    )
    assert summary["rows"] == 2
