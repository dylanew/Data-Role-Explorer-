"""Tests for src/clean.py on small hand-made postings, so they run in a second."""

import pandas as pd
import pytest

from src import clean


def posting(**overrides):
    """One raw row in the shape load_raw() returns, with sensible defaults."""
    row = {
        "id": "1",
        "title_raw": "Data Scientist",
        "company": "Acme",
        "location": "London",
        "region": "London",
        "latitude": None,
        "longitude": None,
        "category": "IT Jobs",
        "contract_type": None,
        "contract_time": None,
        "created": pd.Timestamp("2026-10-01", tz="UTC"),
        "description_raw": "Build models.",
        "salary_min_raw": 50000.0,
        "salary_max_raw": 60000.0,
        "salary_is_predicted": False,
        "search_term": "data scientist",
        "redirect_url": "https://example.com",
    }
    row.update(overrides)
    return row


def run(*rows):
    df, counts = clean.clean(pd.DataFrame(rows))
    return df, counts


def test_clean_text_strips_html_and_entities():
    assert clean.clean_text("<p>Python &amp; SQL</p>&xa;&xa; skills…") == "Python & SQL skills"


@pytest.mark.parametrize("raw, expected", [
    ("Senior ML Engineer - Fulfillment (Hybrid)", "senior machine learning engineer"),
    ("BI Analyst", "business intelligence analyst"),
    ("Power BI Developer", "power bi developer"),
    ("Data Scientist III", "data scientist"),
    ("Data Analyst (Whole Family Support) - NAY18016", "data analyst"),
    ("Senior Manager, Data Science", "senior manager data science"),
    ("Sr. Data Engineer | Remote", "senior data engineer"),
])
def test_normalise_title(raw, expected):
    assert clean.normalise_title(raw) == expected


@pytest.mark.parametrize("title, family", [
    ("Senior Data Engineer", "data engineer"),
    ("ML Engineer", "machine learning engineer"),
    ("Machine Learning Research Engineer", "machine learning engineer"),
    ("Analytics Engineer", "analytics engineer"),
    ("Lead Data Scientist", "data scientist"),
    ("BI Analyst", "bi analyst"),
    ("Insights Analyst", "data analyst"),
    ("Senior AI Engineer", "ai engineer"),
    ("Software Engineer", "other"),
])
def test_title_family(title, family):
    cleaned = clean.normalise_title(title)
    assert clean.first_match(cleaned, clean.FAMILY_RULES, clean.DEFAULT_FAMILY) == family


@pytest.mark.parametrize("title, level", [
    ("Senior / Lead AI Engineer", "lead"),
    ("Graduate Data Analyst", "entry"),
    ("Head of Data", "head"),
    ("Data Engineer", "unspecified"),
])
def test_seniority(title, level):
    assert clean.first_match(title, clean.SENIORITY_RULES, clean.DEFAULT_SENIORITY) == level


def test_same_id_is_merged_and_keeps_all_search_terms():
    df, counts = run(posting(search_term="data scientist"),
                     posting(search_term="ml engineer"))
    assert counts["after_merging_same_id"] == 1
    assert df.loc[0, "search_terms"] == "data scientist;ml engineer"
    assert df.loc[0, "n_search_terms"] == 2


def test_reposts_in_other_towns_are_merged():
    df, _ = run(posting(id="1", location="Leeds", search_term="data analyst"),
                posting(id="2", location="York", search_term="bi analyst",
                        created=pd.Timestamp("2026-09-01", tz="UTC")))
    assert len(df) == 1
    assert df.loc[0, "id"] == "2"          # earliest copy kept
    assert df.loc[0, "n_locations"] == 2
    assert df.loc[0, "search_terms"] == "bi analyst;data analyst"


def test_different_jobs_with_same_boilerplate_are_kept():
    df, _ = run(posting(id="1", title_raw="Data Scientist"),
                posting(id="2", title_raw="Senior Data Scientist"))
    assert len(df) == 2


def test_salary_rules():
    # Different descriptions so these aren't merged as reposts of one ad.
    rows = [
        posting(id="1"),                                                  # advertised
        posting(id="2", salary_min_raw=500.0, salary_max_raw=600.0),      # day rate
        posting(id="3", salary_min_raw=45000.0, salary_max_raw=45000.0,
                salary_is_predicted=True),                                # Adzuna estimate
        posting(id="4", salary_min_raw=None, salary_max_raw=None),        # none
        posting(id="5", salary_min_raw=40000.0, salary_max_raw=None),     # no max
    ]
    df, _ = run(*[{**r, "description_raw": r["id"]} for r in rows])
    df = df.set_index("id")
    assert df.loc["1", "salary_mid"] == 55000
    assert df.loc["1", "salary_source"] == "advertised"
    assert pd.isna(df.loc["2", "salary_mid"])
    assert df.loc["2", "salary_source"] == "dropped_rate"
    assert pd.isna(df.loc["3", "salary_mid"])
    assert df.loc["3", "salary_estimate"] == 45000
    assert df.loc["4", "salary_source"] == "missing"
    assert df.loc["5", "salary_mid"] == 40000


def test_flags():
    df, _ = run(posting(id="1", title_raw="Software Engineer", description_raw="a"),
                posting(id="2", title_raw="Data Analyst Placement Programme",
                        description_raw="b"),
                posting(id="3", title_raw="Data Analyst", description_raw="c"))
    df = df.set_index("id")
    assert not df.loc["1", "is_data_role"]
    assert df.loc["2", "is_training_ad"]
    assert df.loc["3", "is_data_role"] and not df.loc["3", "is_training_ad"]
