"""Tests for src/collect.py that use a fake API, so no keys or network are needed."""

import json
from unittest.mock import MagicMock

import pytest

from src import collect


def fake_response(status, payload=None):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload or {}
    if status >= 400:
        response.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return response


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(collect.time, "sleep", lambda seconds: None)


def test_collect_term_stops_when_all_results_seen():
    session = MagicMock()
    session.get.side_effect = [
        fake_response(200, {"count": 3, "results": [{"id": "1"}, {"id": "2"}]}),
        fake_response(200, {"count": 3, "results": [{"id": "3"}]}),
    ]
    postings, total = collect.collect_term(session, "gb", "data analyst", 10, "id", "key")

    assert total == 3
    assert [p["id"] for p in postings] == ["1", "2", "3"]
    assert all(p["search_term"] == "data analyst" for p in postings)
    assert session.get.call_count == 2


def test_collect_term_respects_page_limit():
    session = MagicMock()
    session.get.return_value = fake_response(200, {"count": 1000, "results": [{"id": "x"}]})
    collect.collect_term(session, "gb", "data engineer", 3, "id", "key")
    assert session.get.call_count == 3


def test_fetch_page_retries_on_rate_limit():
    session = MagicMock()
    session.get.side_effect = [fake_response(429), fake_response(200, {"results": []})]
    data = collect.fetch_page(session, "gb", "data scientist", 1, "id", "key")
    assert data == {"results": []}
    assert session.get.call_count == 2


def test_fetch_page_fails_fast_on_bad_credentials():
    session = MagicMock()
    session.get.return_value = fake_response(401)
    with pytest.raises(Exception):
        collect.fetch_page(session, "gb", "data scientist", 1, "id", "key")
    assert session.get.call_count == 1


def test_save_snapshot_writes_jsonl_and_metadata(tmp_path):
    postings = [{"id": "1", "title": "Data Analyst"}, {"id": "2", "title": "Data Engineer"}]
    data_path, meta_path = collect.save_snapshot(postings, {"total_postings": 2}, "gb", tmp_path)

    lines = data_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["1", "2"]
    assert json.loads(meta_path.read_text())["total_postings"] == 2
    assert data_path.name.startswith("adzuna_gb_")
