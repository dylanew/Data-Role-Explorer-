"""Tests for the match-me scoring in src/match.py."""

import io

import numpy as np
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


# --- Embedding side (no real model: a fake one that records what it was given) -----

class FakeModel:
    def __init__(self):
        self.seen = []

    def encode(self, texts, normalize_embeddings):
        self.seen.extend(texts)
        # "pipelines" points one way, everything else the other.
        return np.array([[1.0, 0.0] if "pipelines" in t else [0.0, 1.0] for t in texts])


def test_role_centres_are_unit_length_means():
    features = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    centres = match.role_centres(features, np.array([1, 1, 2]), role_clusters=(1, 2))
    assert np.allclose(centres[1], [2 ** -0.5, 2 ** -0.5])
    assert np.allclose(centres[2], [0.0, 1.0])


def test_chunk_text_packs_sentences_up_to_size():
    text = "One two. Three four.\nFive six."
    assert match.chunk_text(text, size=20) == ["One two. Three four.", "Five six."]
    assert match.chunk_text(text, size=1000) == ["One two. Three four. Five six."]
    assert match.chunk_text("   ") == []


def test_embed_text_masks_role_names_and_averages_chunks():
    model = FakeModel()
    vector = match.embed_text("Data Engineer. Built pipelines.\nLed a team.", model)
    assert all("engineer" not in t.lower() for t in model.seen)
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert match.embed_text("Data Engineer", FakeModel()) is None  # nothing left after masking


def test_closest_roles_and_close_call():
    centres = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
    roles = match.closest_roles(np.array([0.6, 0.8]), centres, {1: "Eng", 2: "BI"})
    assert roles["cluster"].tolist() == [2, 1]
    assert not match.is_close_call(roles)
    roles = match.closest_roles(np.array([0.70, 0.71]), centres, {1: "Eng", 2: "BI"})
    assert match.is_close_call(roles)


def minimal_pdf(text):
    """A one-page PDF with text, written by hand so the test needs no PDF writer."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
               b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
               b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
              % (len(objects) + 1, xref))
    out.seek(0)
    return out


def test_pdf_text_reads_the_text_layer():
    pytest.importorskip("pypdf")
    assert "Python and SQL" in match.pdf_text(minimal_pdf("Python and SQL"))
