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


def test_map_groups_role_type_keeps_role_colours_and_greys_out_the_rest():
    from src.cluster import CLUSTER_NAMES, NOT_STATED, ROLE_CLUSTERS

    role = CLUSTER_NAMES[ROLE_CLUSTERS[0]]
    table = pd.DataFrame({"cluster": [ROLE_CLUSTERS[0], 0, 0],
                          "role_type": [role, role, NOT_STATED]})
    groups = {name: (rows.sum(), colour)
              for name, rows, colour, _, _ in projection.map_groups(table, "role_type")}
    # The employer-group ad with a role from its skills is counted under that role.
    assert groups[role] == (2, projection.CLUSTER_COLOURS[ROLE_CLUSTERS[0]])
    assert groups[NOT_STATED] == (1, projection.NOT_STATED_COLOUR)


def test_map_groups_cluster_has_one_group_per_cluster():
    table = pd.DataFrame({"cluster": [0, 3], "role_type": ["a", "b"]})
    assert len(list(projection.map_groups(table, "cluster"))) == len(projection.CLUSTER_COLOURS)
