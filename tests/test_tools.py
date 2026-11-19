"""Offline tests for the fixed analysis tools. No API key and no Drive access needed."""

import base64
import json

import pytest

import main


CSV_A = "\n".join(
    [
        "region,product,units,revenue,sale_date",
        "north,widget,10,250.50,2024-01-05",
        "north,gadget,5,300,2024-01-07",
        "south,widget,7,175,2024-01-09",
        "south,gadget,3,180,2024-01-11",
        "east,widget,20,500,2024-02-01",
        "east,gadget,2,120,2024-02-03",
    ]
)


@pytest.fixture()
def loaded_dataset(monkeypatch):
    async def fake_call_drive_tool(tool_name, arguments):
        if tool_name == main.METADATA_TOOL:
            return {"id": "FILE1234567890AB", "title": "sales.csv", "mime_type": "text/csv"}
        if tool_name == main.DOWNLOAD_TOOL:
            return {"content": base64.b64encode(CSV_A.encode()).decode()}
        raise AssertionError(tool_name)

    monkeypatch.setattr(main, "_call_drive_tool", fake_call_drive_tool)
    main._DATASETS.clear()
    main._CURRENT = None
    return main.load_dataset


@pytest.mark.asyncio()
async def test_load_dataset_reports_schema(loaded_dataset):
    result = await loaded_dataset(drive_link="https://drive.google.com/file/d/FILE1234567890AB/view")
    assert result["rows"] == 6
    assert result["title"] == "sales.csv"
    columns = {c["name"]: c for c in result["columns"]}
    assert columns["units"]["dtype"] == "int64"
    assert columns["revenue"]["dtype"] == "float64"
    assert columns["region"]["dtype"] == "object"
    assert columns["sale_date"]["dtype"].startswith("datetime64")


@pytest.mark.asyncio()
async def test_describe_columns_statistics(loaded_dataset):
    await loaded_dataset(drive_link="FILE1234567890AB")
    entries = {e["column"]: e for e in await main.describe_columns()}
    assert entries["units"]["kind"] == "numeric"
    assert entries["units"]["min"] == 2
    assert entries["units"]["max"] == 20
    assert entries["region"]["kind"] == "text"
    assert set(entries["region"]["distinct_values"]) == {"north", "south", "east"}


@pytest.mark.asyncio()
async def test_filter_rows_equals_counts(loaded_dataset):
    await loaded_dataset(drive_link="FILE1234567890AB")
    result = await main.filter_rows(column="Region", operation="equals", value="north")
    assert result["total_matches"] == 2
    assert result["total_rows"] == 6
    assert len(result["rows"]) == 2
    assert {r["product"] for r in result["rows"]} == {"widget", "gadget"}


@pytest.mark.asyncio()
async def test_filter_rows_numeric_and_between(loaded_dataset):
    await loaded_dataset(drive_link="FILE1234567890AB")
    result = await main.filter_rows(column="units", operation="greater_than", value="6")
    assert result["total_matches"] == 3
    between = await main.filter_rows(column="revenue", operation="between", value=176, value_2=301)
    assert between["total_matches"] == 3
    has_null = await main.filter_rows(column="units", operation="in", value_list=["10", "3"])
    assert has_null["total_matches"] == 2


@pytest.mark.asyncio()
async def test_groupby_aggregate_sum_mean(loaded_dataset):
    await loaded_dataset(drive_link="FILE1234567890AB")
    result = await main.groupby_aggregate(
        group_by="region",
        aggregations=[{"column": "revenue", "func": "sum"}, {"column": "units", "func": "mean"}],
        order_by="sum_revenue",
    )
    by_region = {r["region"]: r for r in result["rows"]}
    assert len(result["rows"]) == 3
    assert by_region["north"]["sum_revenue"] == 550.5
    assert by_region["north"]["mean_units"] == 7.5
    assert by_region["south"]["sum_revenue"] == 355
    assert result["total_groups"] == 3
    counts = await main.groupby_aggregate(group_by=["region", "product"], aggregations=[])
    assert counts["total_groups"] == 6
    assert counts["rows"][0]["count_rows"] == 1


@pytest.mark.asyncio()
async def test_errors_are_actionable(monkeypatch):
    async def fake_call_drive_tool(tool_name, arguments):
        if tool_name == main.METADATA_TOOL:
            return {"id": "X12345678901234567", "title": "sales.csv", "mime_type": "application/pdf"}
        raise AssertionError(tool_name)

    monkeypatch.setattr(main, "_call_drive_tool", fake_call_drive_tool)
    main._DATASETS.clear()
    main._CURRENT = None
    with pytest.raises(main.agents.ToolError, match="Only CSV files"):
        await main.load_dataset(drive_link="X12345678901234567")

    with pytest.raises(main.agents.ToolError, match="No dataset is loaded"):
        await main.describe_columns()


def test_agent_turn_with_scripted_model(monkeypatch):
    import asyncio

    import mistralai.agents as sdk_agents
    from mistralai import agents

    async def fake_call_drive_tool(tool_name, arguments):
        if tool_name == main.METADATA_TOOL:
            return {"id": "FILE1234567890AB", "title": "sales.csv", "mime_type": "text/csv"}
        return {"content": base64.b64encode(CSV_A.encode()).decode()}

    monkeypatch.setattr(main, "_call_drive_tool", fake_call_drive_tool)
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-offline-test-dummy")
    agent = main.agent.override(
        model=sdk_agents.testing.ScriptedModel(
            [
                sdk_agents.testing.call("load_dataset", drive_link="https://drive.google.com/file/d/FILE1234567890AB/view"),
                sdk_agents.testing.reply("Loaded sales.csv with 6 rows."),
            ]
        )
    )
    state = asyncio.run(agent.run("Analyze https://drive.google.com/file/d/FILE1234567890AB/view"))
    assert isinstance(state, agents.Completed)
    assert "6 rows" in state.text
    assert [call.name for call in state.tool_calls] == ["client.load_dataset"]
