"""Load the precomputed files the app reads. Nothing heavy happens here.

Every table comes from data/processed, written offline by the src/ pipeline
(cleaning, skills, clustering, UMAP). The app only reads CSVs and does small
pandas summaries, so it starts fast and needs no ML model.

st.cache_data keeps each table in memory after the first load, so moving
between pages does not reread the files.
"""

import json

import pandas as pd
import streamlit as st

from src.cluster import CLUSTER_NAMES, ROLE_CLUSTERS
from src.embed import PROCESSED_DIR, in_scope, load_features
from src.match import load_model, role_centres, role_profiles

# Colour used for single-series charts (one bar colour, nothing to tell apart).
# Kept apart from the cluster colours so no chart seems to be about a cluster.
NEUTRAL = "#5b6b7f"


def snapshot_suffix():
    """'gb_2026-10-24' for the newest clustered snapshot."""
    newest = sorted(PROCESSED_DIR.glob("clusters_*.meta.json"))[-1]
    return newest.name.removeprefix("clusters_").removesuffix(".meta.json")


@st.cache_data
def load(suffix):
    """All tables for one snapshot, as a dict of DataFrames (plus metadata)."""
    def read(name):
        return pd.read_csv(PROCESSED_DIR / f"{name}_{suffix}.csv", dtype={"id": str})

    postings = read("postings")
    postings = in_scope(postings)
    clusters = read("clusters")
    skills = read("skills")
    skills = skills[skills["id"].isin(postings["id"])]
    return {
        "postings": postings,
        "clusters": clusters,
        "skills": skills,
        "projection": read("projection"),
        "profiles": read("cluster_profiles"),
        "role_profiles": role_profiles(skills, clusters, ROLE_CLUSTERS),
        "cluster_meta": json.loads((PROCESSED_DIR / f"clusters_{suffix}.meta.json").read_text()),
        "projection_meta": json.loads(
            (PROCESSED_DIR / f"projection_{suffix}.meta.json").read_text()),
    }


def snapshot_label(suffix):
    country, day = suffix.split("_")
    return f"Adzuna {country.upper()} snapshot of {day}"


def cluster_order():
    """Role types first, then the employer / ad groups, as on the map."""
    return list(ROLE_CLUSTERS) + [c for c in sorted(CLUSTER_NAMES) if c not in ROLE_CLUSTERS]


@st.cache_data
def load_centres(suffix):
    """{cluster: centre} for the role types, from the precomputed ad embeddings.

    Reads the same embeddings file the clusters were built from (named in the
    clusters meta file), so the centres always match the clusters.
    """
    meta = json.loads((PROCESSED_DIR / f"clusters_{suffix}.meta.json").read_text())
    ids, features = load_features(PROCESSED_DIR / meta["features"])
    clusters = pd.read_csv(PROCESSED_DIR / f"clusters_{suffix}.csv", dtype={"id": str})
    labels = clusters.set_index("id").loc[ids, "cluster"].to_numpy()
    return role_centres(features, labels, ROLE_CLUSTERS)


@st.cache_resource(show_spinner="Loading the language model (first visit only)...")
def model():
    """The sentence model, loaded once and shared by every visitor."""
    return load_model()
