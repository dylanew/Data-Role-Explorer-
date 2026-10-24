"""Group postings into role types, and compare the groups with job titles.

Usage (from the repo root):
    python -m src.cluster compare                 # score every feature set x method
    python -m src.cluster final features_tfidf_masked_gb_2026-10-24.npz --kmeans 8
    python -m src.cluster final <features file> --hdbscan 25

Two steps, because the brief asks for a human choice in between:

  compare  Runs K-means (k = 3..12) and HDBSCAN (several minimum cluster
           sizes) on every features_*.npz written by src/embed.py, and writes
           one row per run with its scores. Nothing is chosen here.
  final    Runs the one setting that was chosen, and writes each posting's
           cluster plus a profile of every cluster (top terms, top skills,
           title families, sample titles) so the clusters can be inspected
           and named by hand.

The two methods:
    - K-means: you choose k; every posting goes to the nearest of k centres.
      Simple and stable, but it assumes round clusters of similar size and
      forces every posting into some cluster, even an odd one.
    - HDBSCAN: finds dense regions and how many there are on its own; you
      choose the smallest group worth calling a cluster. Postings that sit
      between groups are labelled noise (-1) instead of being forced in.
      Density does not work well in 100+ dimensions (everything is roughly
      equally far apart), so HDBSCAN runs on a 10-dimensional UMAP
      projection of the features, which is the usual recipe.

The scores, per run:
    - silhouette: for each posting, how much closer it is to its own cluster
      than to the next one, averaged; from -1 to 1, higher is better. It is
      always computed in the original feature space with cosine distance
      (never on the UMAP projection, which would flatter HDBSCAN), and
      without noise points.
    - noise_share: share of postings HDBSCAN left out (0 for K-means).
    - ari_vs_title / nmi_vs_title: how closely the clusters match
      title_family, from 0 (unrelated) to 1 (identical grouping). Adjusted
      Rand index counts pairs of postings put together in both; normalised
      mutual information measures how much knowing one tells you about the
      other. These answer the research question; they are not used to pick
      the "best" run, because the point is to see whether the market's own
      groups differ from the titles.

Outputs, in data/processed/:
    cluster_comparison_<country>_<date>.csv   compare: one row per run
    clusters_<country>_<date>.csv             final: id, cluster, cluster_name
    cluster_profiles_<country>_<date>.csv     final: one row per cluster
    cluster_profiles_<country>_<date>.md      final: the same, readable, with
                                              sample titles for inspection
    clusters_<country>_<date>.meta.json       final: the setting used
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from src.embed import (PROCESSED_DIR, SEED, build_text, in_scope, latest_postings,
                       load_features, make_vectorizer)

K_RANGE = range(3, 13)
MIN_CLUSTER_SIZES = (10, 15, 25, 40)

# UMAP settings for HDBSCAN. n_neighbors 15 is the library default and looks
# at local structure; min_dist 0 lets similar points sit right on top of each
# other, which helps density-based clustering (and is what the UMAP docs
# recommend for clustering, as opposed to drawing).
UMAP_SETTINGS = dict(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine")

# Skills from this group describe how a job is done, not what kind of job it
# is, so they were left out of the features. They are still shown in the
# profiles, where they help describe a cluster.
PROFILE_ONLY_SKILL_GROUP = "business & ways of working"


# --- Running one clustering -------------------------------------------------------

def run_kmeans(features, k):
    """K-means with 10 restarts from different random centres; the best is kept."""
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(features)


def reduce_for_hdbscan(features):
    import umap  # imported here: it is slow to import and only HDBSCAN needs it

    return umap.UMAP(**UMAP_SETTINGS, random_state=SEED).fit_transform(features)


def run_hdbscan(reduced, min_cluster_size):
    return HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(reduced)


def score(features, labels, title_family):
    """Scores for one run (see the module docstring for what they mean)."""
    clustered = labels >= 0
    n_clusters = len(set(labels[clustered]))
    silhouette = (silhouette_score(features[clustered], labels[clustered], metric="cosine")
                  if n_clusters >= 2 else np.nan)
    sizes = np.bincount(labels[clustered]) if n_clusters else np.array([0])
    return {
        "n_clusters": n_clusters,
        "silhouette": round(float(silhouette), 4),
        "noise_share": round(float(1 - clustered.mean()), 4),
        "smallest_cluster": int(sizes.min()),
        "largest_cluster": int(sizes.max()),
        "ari_vs_title": round(adjusted_rand_score(title_family[clustered], labels[clustered]), 4),
        "nmi_vs_title": round(normalized_mutual_info_score(title_family[clustered],
                                                           labels[clustered]), 4),
    }


def describe_file(name):
    """features_emb-all-minilm-l6-v2_masked_gb_2026-10-24.npz -> ('emb-all-minilm-l6-v2', 'masked')."""
    parts = name.removeprefix("features_").split("_")
    return parts[0], parts[1]


def compare(postings, feature_files):
    """Score every (features file, method, setting). One row per run."""
    title_family = postings.set_index("id")["title_family"]
    rows = []
    for path in feature_files:
        ids, features = load_features(path)
        families = title_family.loc[ids].to_numpy()
        method, text = describe_file(path.name)
        print(f"{path.name}: K-means", flush=True)
        for k in K_RANGE:
            labels = run_kmeans(features, k)
            rows.append({"features": method, "text": text, "algorithm": "kmeans",
                         "setting": f"k={k}", **score(features, labels, families)})
        print(f"{path.name}: UMAP + HDBSCAN", flush=True)
        reduced = reduce_for_hdbscan(features)
        for size in MIN_CLUSTER_SIZES:
            labels = run_hdbscan(reduced, size)
            rows.append({"features": method, "text": text, "algorithm": "hdbscan",
                         "setting": f"min_cluster_size={size}",
                         **score(features, labels, families)})
    return pd.DataFrame(rows)


# --- Describing clusters --------------------------------------------------------

def top_terms(texts, labels, n=10):
    """The words that most set each cluster apart from the rest.

    For each word: its average TF-IDF weight inside the cluster minus its
    average weight overall. Plain within-cluster averages would put the same
    common words ("experience", "team") at the top of every cluster.
    """
    vectorizer = make_vectorizer()
    weights = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())
    overall = np.asarray(weights.mean(axis=0)).ravel()
    result = {}
    for cluster in sorted(set(labels)):
        inside = np.asarray(weights[labels == cluster].mean(axis=0)).ravel()
        result[cluster] = list(terms[np.argsort(inside - overall)[::-1][:n]])
    return result


def top_skills(skills, ids, labels, n=8, min_share=0.05):
    """Skills mentioned most often in each cluster's descriptions, with lift.

    share: share of the cluster's postings that mention the skill.
    lift:  share in the cluster / share overall. 2.0 = twice as common here.
    Ranked by share among skills with lift above 1, so each list shows what is
    both common in the cluster and more common than elsewhere.
    """
    hits = skills[skills["in_description"]][["id", "skill", "category"]]
    cluster_of = pd.Series(labels, index=ids, name="cluster")
    hits = hits[hits["id"].isin(cluster_of.index)].join(cluster_of, on="id")
    sizes = cluster_of.value_counts()
    overall_share = hits.groupby("skill")["id"].nunique() / len(cluster_of)

    result = {}
    for cluster, group in hits.groupby("cluster"):
        share = group.groupby("skill")["id"].nunique() / sizes[cluster]
        table = pd.DataFrame({"share": share, "lift": share / overall_share[share.index]})
        table = table[(table["share"] >= min_share) & (table["lift"] > 1)]
        result[cluster] = table.sort_values("share", ascending=False).head(n).round(2)
    return result


def draft_name(terms, skills_table):
    """A first name from the top two skills, or top two terms if no skills.

    Only a starting point: the final names go in CLUSTER_NAMES after reading
    the profiles.
    """
    picks = list(skills_table.index[:2]) if skills_table is not None and len(skills_table) else []
    picks += [t for t in terms if t not in picks][: 2 - len(picks)]
    return " / ".join(picks)


# Final names, written by hand after reading cluster_profiles_*.md. Keys are
# cluster numbers from the final run. Empty until a run has been chosen.
CLUSTER_NAMES = {}


def profiles(postings, skills, ids, labels, texts):
    """One row per cluster: size, names, terms, skills, title families."""
    frame = postings.set_index("id").loc[ids].assign(cluster=labels)
    terms = top_terms(texts, labels)
    skill_tables = top_skills(skills, ids, labels)
    rows = []
    for cluster, group in frame.groupby("cluster"):
        families = group["title_family"].value_counts(normalize=True).round(2)
        table = skill_tables.get(cluster)
        rows.append({
            "cluster": cluster,
            "cluster_name": CLUSTER_NAMES.get(cluster)
                            or ("noise" if cluster == -1 else draft_name(terms[cluster], table)),
            "size": len(group),
            "top_terms": ", ".join(terms[cluster]),
            "top_skills": ", ".join(f"{s} ({r.share:.0%}, x{r.lift:.1f})"
                                    for s, r in table.iterrows()) if table is not None else "",
            "main_title_family": families.index[0],
            "main_title_family_share": families.iloc[0],
            "title_families": "; ".join(f"{f} {s:.0%}" for f, s in families.items()),
            "seniority": "; ".join(f"{f} {s:.0%}" for f, s in
                                   group["seniority"].value_counts(normalize=True).head(3).items()),
            "median_salary": group["salary_mid"].median(),
            "sample_titles": " | ".join(group["title"].sample(min(8, len(group)),
                                                             random_state=SEED)),
        })
    return pd.DataFrame(rows)


def crosstab(postings, ids, labels):
    """Cluster x title family, as shares of each title family.

    Read down a column: where did the data analysts end up?
    """
    families = postings.set_index("id").loc[ids, "title_family"].to_numpy()
    return pd.crosstab(labels, families, normalize="columns").round(2)


def write_markdown(path, profile, table, setting):
    lines = [f"# Cluster profiles\n", f"Setting: `{setting}`\n"]
    for row in profile.itertuples():
        lines += [f"## {row.cluster}: {row.cluster_name} ({row.size} postings)\n",
                  f"- Top terms: {row.top_terms}",
                  f"- Top skills: {row.top_skills or 'none above 5%'}",
                  f"- Title families: {row.title_families}",
                  f"- Seniority: {row.seniority}",
                  f"- Sample titles: {row.sample_titles}\n"]
    lines += ["## Where each title family ended up (share of the family)\n",
              "```", table.to_string(), "```"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# --- Command line ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    cmp_parser = sub.add_parser("compare", help="score every feature set x method")
    cmp_parser.add_argument("--postings", type=Path)
    fin_parser = sub.add_parser("final", help="run one chosen setting and write profiles")
    fin_parser.add_argument("features", help="features_*.npz file name in data/processed")
    fin_parser.add_argument("--postings", type=Path)
    how = fin_parser.add_mutually_exclusive_group(required=True)
    how.add_argument("--kmeans", type=int, metavar="K")
    how.add_argument("--hdbscan", type=int, metavar="MIN_CLUSTER_SIZE")
    args = parser.parse_args()

    path = args.postings or latest_postings()
    postings = in_scope(pd.read_csv(path, dtype={"id": str}))
    suffix = path.stem.replace("postings_", "", 1)
    out_dir = path.parent

    if args.step == "compare":
        files = sorted(out_dir.glob(f"features_*_{suffix}.npz"))
        if not files:
            raise SystemExit("No feature files. Run python -m src.embed first.")
        result = compare(postings, files)
        out = out_dir / f"cluster_comparison_{suffix}.csv"
        result.to_csv(out, index=False)
        print(result.to_string(index=False))
        print(f"\nwrote {out.name}")
        return

    ids, features = load_features(out_dir / args.features)
    _, text_version = describe_file(args.features)
    if args.kmeans:
        labels, setting = run_kmeans(features, args.kmeans), f"kmeans k={args.kmeans}"
    else:
        labels = run_hdbscan(reduce_for_hdbscan(features), args.hdbscan)
        setting = f"hdbscan min_cluster_size={args.hdbscan}"

    scoped = postings.set_index("id").loc[ids].reset_index()
    skills = pd.read_csv(out_dir / f"skills_{suffix}.csv", dtype={"id": str})
    profile = profiles(postings, skills, ids, labels, build_text(scoped, text_version))
    table = crosstab(postings, ids, labels)
    names = profile.set_index("cluster")["cluster_name"]

    pd.DataFrame({"id": ids, "cluster": labels, "cluster_name": names.loc[labels].to_numpy()}
                 ).to_csv(out_dir / f"clusters_{suffix}.csv", index=False)
    profile.to_csv(out_dir / f"cluster_profiles_{suffix}.csv", index=False)
    write_markdown(out_dir / f"cluster_profiles_{suffix}.md", profile, table,
                   f"{args.features}, {setting}")
    meta = {"source_postings": path.name, "features": args.features, "setting": setting,
            "seed": SEED, "umap": UMAP_SETTINGS if args.hdbscan else None,
            **score(features, labels, scoped["title_family"].to_numpy())}
    (out_dir / f"clusters_{suffix}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(profile[["cluster", "cluster_name", "size", "main_title_family",
                   "main_title_family_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
