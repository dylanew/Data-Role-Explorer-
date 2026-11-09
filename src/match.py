"""Match a visitor's skills or CV to a role type (the "match me" page, milestone 6).

Hybrid approach: sentence embeddings pick the role, the skill dictionary says
what is missing.

1. Pick the role (embeddings). The visitor's text is embedded with the same
   model as the job ads (MiniLM), and compared with the centre of each of
   the three role-type clusters. The closest centre is the match. This
   reads the whole text, so "built nightly ingestion jobs in Airflow" counts
   towards data platform work even though "data pipelines" is never said.
2. Find skills (dictionary). The same dictionary and rules used on the ads
   (src/skills.py), so "sklearn" in a CV counts exactly as it does in an ad.
3. Skill profile per role type: for every skill, the share of that role's
   ads that mention it, keeping skills at least MIN_SHARE of them mention.
4. Skills to add: the matched role's profile skills the visitor does not
   have, most asked-for first. Coverage (the profile shares of the skills
   you have, over the total) is shown too, as a second opinion.

Design notes:
    - Only the three role types are matched (src.cluster.ROLE_CLUSTERS). The
      other five clusters group ads by employer or ad template; "you match
      'Banking employers'" would not tell anyone anything about their skills.
    - The visitor's text gets the same role-name masking as the ads
      (src.embed.mask_roles). The centres were built from masked ads, so a
      CV saying "Data Engineer" must not get a head start from the title.
      The ads' boilerplate filter is not applied: it was written for job
      ads, and would drop CV lines about "global" teams or "Office".
    - The ads are ~500-character snippets, and MiniLM only reads the first
      256 word pieces of a text. A CV is much longer, so it is cut into
      ~CHUNK_CHARS pieces (on sentence or line breaks), each piece is
      embedded, and the pieces are averaged. Every part of the CV counts,
      and each piece looks like what the model saw for the ads.
    - Closeness is cosine similarity to a centre (the mean of the role's ad
      embeddings, scaled to length 1). On its own a number like 0.42 means
      little, and a CV is never as close to a centre as the ads themselves
      (it is written differently), so the app compares the roles with each
      other: if the top two are within CLOSE_CALL, it says it is a close call
      and shows both. 0.03 is a judgement call from trying sample CVs, not a
      measured threshold.
    - Skill profiles use only ads in the role's own cluster, not ads that got
      the role type from their skills (role_type_source == "skills"). Those
      were labelled with these same skills, so counting them would be circular.
    - The heavy part (embedding 1,857 ads) was done offline by src/embed.py;
      the app only embeds the one text it is given. The model is imported
      lazily so the rest of this module works without torch.
"""

import re

import numpy as np
import pandas as pd

from src.embed import DEFAULT_MODEL, SENTENCE_BREAK, mask_roles
from src.skills import find_skills

# Same threshold as src.cluster.signature_skills: a skill must appear in at
# least 2% of a role's ads to be part of its profile.
MIN_SHARE = 0.02

# About the length of an Adzuna snippet, and well inside MiniLM's 256 word pieces.
CHUNK_CHARS = 500

# Top two roles closer than this in cosine similarity = "close call" (see above).
CLOSE_CALL = 0.03


def role_profiles(skills, clusters, role_clusters, min_share=MIN_SHARE):
    """One row per (role cluster, skill) in a profile.

    skills:   the skills_<...>.csv table (one row per posting and skill).
    clusters: the clusters_<...>.csv table (one row per posting).
    Returns columns cluster, cluster_name, skill, category, share, lift, where
    lift is how many times more common the skill is in this role than across
    all postings.
    """
    in_scope = clusters[["id", "cluster", "cluster_name"]]
    hits = skills[skills["in_description"]].merge(in_scope, on="id")
    overall = hits.groupby("skill")["id"].nunique() / len(in_scope)
    category = hits.drop_duplicates("skill").set_index("skill")["category"]

    rows = []
    for cluster in role_clusters:
        members = in_scope[in_scope["cluster"] == cluster]
        share = hits[hits["cluster"] == cluster].groupby("skill")["id"].nunique() / len(members)
        share = share[share >= min_share].sort_values(ascending=False)
        for skill, value in share.items():
            rows.append({"cluster": cluster, "cluster_name": members["cluster_name"].iloc[0],
                         "skill": skill, "category": category[skill],
                         "share": round(value, 4), "lift": round(value / overall[skill], 2)})
    return pd.DataFrame(rows, columns=["cluster", "cluster_name", "skill", "category",
                                       "share", "lift"])


def skills_in_text(text):
    """The dictionary skills found in a piece of text, as a sorted list."""
    return sorted(find_skills(text))


def score_roles(my_skills, profiles):
    """Coverage of each role's profile by my_skills, best match first.

    Returns one row per role: cluster, cluster_name, coverage (0 to 1),
    n_have and n_profile (how many profile skills you have, out of how many).
    """
    mine = set(my_skills)
    rows = []
    for (cluster, name), profile in profiles.groupby(["cluster", "cluster_name"], sort=False):
        have = profile["skill"].isin(mine)
        rows.append({"cluster": cluster, "cluster_name": name,
                     "coverage": profile.loc[have, "share"].sum() / profile["share"].sum(),
                     "n_have": int(have.sum()), "n_profile": len(profile)})
    return pd.DataFrame(rows).sort_values("coverage", ascending=False, kind="stable") \
                             .reset_index(drop=True)


def missing_skills(my_skills, profiles, cluster):
    """The role's profile skills that my_skills does not include, most asked-for first."""
    profile = profiles[profiles["cluster"] == cluster]
    return profile[~profile["skill"].isin(set(my_skills))].reset_index(drop=True)


# --- Embedding side: pick the role --------------------------------------------

def role_centres(features, labels, role_clusters):
    """{cluster: centre}, where a centre is the mean ad embedding, scaled to length 1."""
    centres = {}
    for cluster in role_clusters:
        mean = features[labels == cluster].mean(axis=0)
        centres[cluster] = mean / np.linalg.norm(mean)
    return centres


def chunk_text(text, size=CHUNK_CHARS):
    """Split text into pieces of up to ~size characters, on sentence or line breaks.

    A single sentence longer than size becomes a piece of its own (the model
    then reads its first 256 word pieces).
    """
    pieces = [p.strip() for line in str(text).splitlines()
              for p in SENTENCE_BREAK.split(line) if p.strip()]
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > size:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def load_model(model_name=DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def embed_text(text, model):
    """One length-1 vector for a whole text: role names masked, chunked, averaged."""
    chunks = [c for c in (mask_roles(chunk) for chunk in chunk_text(text)) if c]
    if not chunks:
        return None
    vectors = model.encode(chunks, normalize_embeddings=True)
    mean = np.asarray(vectors, dtype=np.float32).mean(axis=0)
    return mean / np.linalg.norm(mean)


def closest_roles(vector, centres, names):
    """Cosine similarity of vector to each role centre, closest first."""
    rows = [{"cluster": cluster, "cluster_name": names[cluster],
             "similarity": float(vector @ centre)}
            for cluster, centre in centres.items()]
    return pd.DataFrame(rows).sort_values("similarity", ascending=False, kind="stable") \
                             .reset_index(drop=True)


def is_close_call(roles, margin=CLOSE_CALL):
    """True when the top two roles from closest_roles are within margin."""
    return len(roles) > 1 and roles["similarity"].iloc[0] - roles["similarity"].iloc[1] < margin


# --- Reading a CV -------------------------------------------------------------------

def pdf_text(file):
    """All the text in a PDF (a path or file-like object), page by page.

    Returns "" for a scanned PDF with no text layer; reading images would
    need OCR, which is too heavy for the app.
    """
    from pypdf import PdfReader

    pages = (page.extract_text() or "" for page in PdfReader(file).pages)
    return re.sub(r"[ \t]+", " ", "\n".join(pages)).strip()
