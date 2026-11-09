"""Tests for the 2D map helpers in src/projection.py (UMAP itself is not run)."""

import numpy as np
import pandas as pd

from src import projection


def test_main_view_leaves_out_far_island():
    rng = np.random.default_rng(0)
    body = rng.normal(0, 1, size=(200, 2))
    island = rng.normal(0, 0.1, size=(20, 2)) + [30, 0]
    inside = projection.main_view(np.vstack([body, island]))
    assert inside[:200].mean() > 0.95
    assert not inside[200:].any()


def test_neighbour_cluster_agreement_is_one_for_separate_groups():
    points = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], dtype=float)
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert projection.neighbour_cluster_agreement(points, labels, "euclidean", k=2) == 1.0


def test_build_table_keeps_feature_order_and_adds_details():
    ids = np.array(["b", "a"])
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    clusters = pd.DataFrame({"id": ["a", "b"], "cluster": [3, 5]})
    postings = pd.DataFrame({"id": ["a", "b"], "title": ["A", "B"], "company": ["x", "y"],
                             "location": ["UK", "UK"], "seniority": ["senior", "lead"],
                             "salary_mid": [50000, None]})
    table = projection.build_table(ids, coords, clusters, postings)
    assert list(table["id"]) == ["b", "a"]
    assert list(table["cluster"]) == [5, 3]
    assert list(table["title"]) == ["B", "A"]
    assert table["in_main_view"].all()
