"""Tests for src/skills.py on short hand-written snippets."""

import pandas as pd
import pytest

from src import skills


def found(text):
    return set(skills.find_skills(text))


@pytest.mark.parametrize("text, skill", [
    ("Experience with scikit-learn", "scikit-learn"),
    ("Experience with sklearn", "scikit-learn"),
    ("Experience with scikit learn", "scikit-learn"),
    ("Strong Power BI and DAX", "Power BI"),
    ("Strong power-bi", "Power BI"),
    ("Modern C++ and C#", "C++"),
    ("Modern C++ and C#", "C#"),
    ("Writing T-SQL procedures", "SQL"),
    ("Strong Python or R capability", "R"),
    ("coding (e.g., Python, R, SQL)", "R"),
    ("Advanced Excel skills", "Excel"),
    ("Building RAG systems", "RAG"),
    ("ETL in Microsoft Fabric", "Microsoft Fabric"),
])
def test_synonyms_match(text, skill):
    assert skill in found(text)


@pytest.mark.parametrize("text, skill", [
    ("Build scalable systems", "Scala"),
    ("An excellent opportunity", "Excel"),
    ("You will excel in this role", "Excel"),
    ("A small AI/ML R&D company", "R"),
    ("From A&R to finance", "R"),
    ("Job ID: R-2079", "R"),
    ("Report RAG status weekly", "RAG"),   # red/amber/green, not AI
    ("Ready to go live", "Go"),
    ("The fabric of our business", "Microsoft Fabric"),
    ("Strong JavaScript", "Java"),
])
def test_lookalikes_do_not_match(text, skill):
    assert skill not in found(text)


def test_find_skills_handles_missing_text():
    assert skills.find_skills(None) == {}
    assert skills.find_skills("") == {}


def test_every_skill_has_a_way_to_match():
    for skill, _category, patterns in skills.COMPILED_SKILLS:
        assert patterns, f"{skill} has no synonyms and no special pattern"


def make_postings():
    return pd.DataFrame([
        {"id": "1", "title": "Data Scientist", "title_family": "data scientist",
         "description": "Python, SQL and machine learning.",
         "is_data_role": True, "is_training_ad": False},
        {"id": "2", "title": "Machine Learning Engineer",
         "title_family": "machine learning engineer",
         "description": "Join our friendly team.",
         "is_data_role": True, "is_training_ad": False},
        {"id": "3", "title": "Data Analyst", "title_family": "data analyst",
         "description": "SQL and Excel reporting.",
         "is_data_role": True, "is_training_ad": False},
        {"id": "4", "title": "Data Analyst Placement Programme",
         "title_family": "data analyst", "description": "Learn SQL with us.",
         "is_data_role": True, "is_training_ad": True},
    ])


def test_extract_keeps_title_and_description_separate():
    matches = skills.extract(make_postings())
    ml = matches[(matches["id"] == "2") & (matches["skill"] == "Machine learning")]
    assert len(ml) == 1
    assert ml["in_title"].item() and not ml["in_description"].item()


def test_frequency_counts_descriptions_of_in_scope_postings_only():
    postings = make_postings()
    freq = skills.frequency(postings, skills.extract(postings)).set_index("skill")
    # Posting 4 is a training ad, so SQL is counted twice, not three times.
    assert freq.loc["SQL", "n_postings"] == 2
    assert freq.loc["SQL", "share"] == round(2 / 3, 4)
    assert freq.loc["SQL", "share_data_analyst"] == 1.0
    # Posting 2 mentions machine learning only in its title.
    assert freq.loc["Machine learning", "n_postings"] == 1


def test_summary_counts_postings_with_zero_skills():
    postings = make_postings()
    summary = skills.summarise(postings, skills.extract(postings))
    assert summary["postings_with_zero_skills"] == 1
    assert summary["postings_with_zero_skills_incl_title"] == 0
