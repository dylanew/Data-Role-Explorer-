"""Tests for the match-me scoring in src/match.py."""

import pandas as pd
import pytest

from src import match


def tiny_data():
    # Cluster 1: 4 ads, 2 mention Python, 1 mentions SQL.
    # Cluster 2: 2 ads, both mention Power BI.
    clusters = pd.DataFrame({
        "id": ["a", "b", "c", "d", "e", "f"],
        "cluster": [1, 1, 1, 1, 2, 2],
        "cluster_name": ["Eng"] * 4 + ["BI"] * 2,
    })
    skills = pd.DataFrame({
        "id": ["a", "b", "c", "e", "f", "f"],
        "skill": ["Python", "Python", "SQL", "Power BI", "Power BI", "SQL"],
        "category": ["programming", "programming", "programming", "bi", "bi", "programming"],
        "in_description": [True, True, True, True, True, False],
    })
    return skills, clusters


def test_profiles_are_shares_of_the_roles_ads():
    skills, clusters = tiny_data()
    profiles = match.role_profiles(skills, clusters, role_clusters=(1, 2))
    eng = profiles[profiles["cluster"] == 1].set_index("skill")["share"]
    assert eng.to_dict() == {"Python": 0.5, "SQL": 0.25}
    # Title-only matches (in_description False) do not count.
    bi = profiles[profiles["cluster"] == 2]
    assert bi["skill"].tolist() == ["Power BI"]
    # Power BI is in 2 of 6 ads overall but 2 of 2 BI ads: lift 3.
    assert bi["lift"].iloc[0] == 3.0


def test_min_share_drops_rare_skills():
    skills, clusters = tiny_data()
    profiles = match.role_profiles(skills, clusters, role_clusters=(1,), min_share=0.3)
    assert profiles["skill"].tolist() == ["Python"]


def test_coverage_weights_skills_by_share():
    skills, clusters = tiny_data()
    profiles = match.role_profiles(skills, clusters, role_clusters=(1, 2))
    scores = match.score_roles(["SQL"], profiles).set_index("cluster")
    # SQL is 0.25 of Eng's 0.75 total share.
    assert scores.loc[1, "coverage"] == pytest.approx(1 / 3)
    assert scores.loc[2, "coverage"] == 0
    assert scores.loc[1, "n_have"] == 1 and scores.loc[1, "n_profile"] == 2


def test_best_match_comes_first():
    skills, clusters = tiny_data()
    profiles = match.role_profiles(skills, clusters, role_clusters=(1, 2))
    assert match.score_roles(["Power BI"], profiles)["cluster"].iloc[0] == 2


def test_missing_skills_are_ordered_by_share():
    skills, clusters = tiny_data()
    profiles = match.role_profiles(skills, clusters, role_clusters=(1, 2))
    assert match.missing_skills([], profiles, 1)["skill"].tolist() == ["Python", "SQL"]
    assert match.missing_skills(["Python"], profiles, 1)["skill"].tolist() == ["SQL"]


def test_skills_in_text_uses_the_ad_dictionary():
    found = match.skills_in_text("I use sklearn, Power-BI and Python daily.")
    assert {"scikit-learn", "Power BI", "Python"} <= set(found)
