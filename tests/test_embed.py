"""Tests for the text preparation in src/embed.py and scoring in src/cluster.py."""

import numpy as np

from src import cluster, embed


def test_mask_roles_removes_role_names_only():
    text = "Senior Data Engineer to build data pipelines with other engineers."
    masked = embed.mask_roles(text)
    assert "Engineer" not in masked and "engineers" not in masked
    assert "data pipelines" in masked


def test_drop_boilerplate_keeps_work_sentences():
    text = ("Our mission is to change the world. You will build dashboards in Power BI. "
            "Salary £50,000 plus benefits.")
    assert embed.drop_boilerplate(text) == "You will build dashboards in Power BI."


def test_score_ignores_noise_points():
    features = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [0.7, 0.7]])
    labels = np.array([0, 0, 1, 1, -1])
    families = np.array(["a", "a", "b", "b", "a"])
    result = cluster.score(features, labels, families)
    assert result["n_clusters"] == 2
    assert result["noise_share"] == 0.2
    assert result["ari_vs_title"] == 1.0
