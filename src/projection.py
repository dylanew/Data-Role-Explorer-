"""Project the postings to 2D with UMAP, for the cluster map (milestone 5).

Usage (from the repo root):
    python -m src.projection                          # chosen settings, writes the map data
    python -m src.projection --n-neighbors 30 --min-dist 0.3   # try other settings
    python -m src.projection --preview                # also write an HTML preview of the map

What this does and does not do:
    The clusters were found in the full 384-dimensional embedding space
    (src/cluster.py). This step only squeezes those same embeddings into two
    dimensions so they can be drawn. It does not change any cluster label.
    So two clusters can overlap on the map and still be separate in 384-d,
    and the map's distances between far-apart groups mean little: UMAP keeps
    neighbourhoods (who sits next to whom), not global distances.

It reads the features file and the clusters file named in
clusters_<country>_<date>.meta.json, so the map always uses the same
embeddings the clusters came from.

UMAP settings (see UMAP_2D_SETTINGS below):
    - n_neighbors: how many neighbours each posting looks at when UMAP builds
      its picture of the data. Small values keep fine local detail and can
      shatter the map into many islands; large values keep more of the big
      picture. 15 is the library default; 30 was chosen after comparing
      15/30/50 (see the README): clusters show up a little more cleanly on
      the map for a small loss of trustworthiness.
    - min_dist: how tightly points may be packed together on the map. 0 packs
      similar postings on top of each other (good for clustering, which is
      why src/cluster.py used 0); 0.1 to 0.5 spreads them out so individual
      points stay visible and hoverable, which is what a drawing needs.
    - metric "cosine": the same distance used everywhere else for embeddings.
    - init "pca": where the points start before UMAP moves them. The default
      ("spectral") is not reproducible with umap-learn 0.5.12 and SciPy 1.18:
      even with a fixed seed it gave a different map on every run (checked on
      2026-11-09). Starting from the first two principal components is
      deterministic, and Kobak & Linderman (2021, "Initialization is critical
      for preserving global data structure in both t-SNE and UMAP") found an
      informative start like PCA keeps the big picture about as well.
    - random_state: fixed seed, so the map is the same every run. (UMAP then
      runs single-threaded, which is slower but reproducible.)

Quality check, written to the meta file:
    - trustworthiness (scikit-learn): of each posting's 10 nearest neighbours
      on the map, how many were also near it in 384-d. 1.0 = the map never
      puts strangers next to each other; above ~0.9 is usually good.
    - neighbour_cluster_agreement: share of each posting's 10 nearest map
      neighbours that are in the same cluster, averaged. It says how cleanly
      the clusters show up on the map, compared with the same number in 384-d.

Outputs, in data/processed/:
    projection_<country>_<date>.csv        one row per posting: id, x, y,
                                           in_main_view, the three labels and
                                           hover details. This
                                           is all the app needs for the map.
    projection_<country>_<date>.meta.json  settings and quality scores
    cluster_map_<country>_<date>.html      only with --preview
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

from src.cluster import CLUSTER_NAMES, ROLE_CLUSTERS
from src.embed import PROCESSED_DIR, SEED, latest_postings, load_features

UMAP_2D_SETTINGS = dict(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine",
                        init="pca")
N_CHECK_NEIGHBOURS = 10

# A few employers post many near-identical ads (one recruiter's template, or
# the same company blurb on every job). UMAP draws each of those as a tight
# island far from everything else, which squashes the rest of the map into a
# corner. The map opens zoomed on the main body instead: postings closer to
# the centre than MAIN_VIEW_FACTOR x the median distance to the centre. The
# islands are still drawn and still in the data; zooming out shows them.
MAIN_VIEW_FACTOR = 3

# Details shown when hovering over a posting on the map.
HOVER_COLUMNS = ["title", "company", "location", "seniority", "salary_mid"]


# --- Projection ---------------------------------------------------------------------

def project_2d(features, n_neighbors, min_dist):
    import umap  # imported here: it is slow to import

    settings = {**UMAP_2D_SETTINGS, "n_neighbors": n_neighbors, "min_dist": min_dist}
    return umap.UMAP(**settings, random_state=SEED).fit_transform(features)


def neighbour_cluster_agreement(points, labels, metric, k=N_CHECK_NEIGHBOURS):
    """Average share of each posting's k nearest neighbours that share its cluster."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(points)
    _, idx = nn.kneighbors(points)
    neighbours = labels[idx[:, 1:]]  # column 0 is the posting itself
    return float((neighbours == labels[:, None]).mean())


def quality(features, coords, labels):
    return {
        "trustworthiness": round(float(trustworthiness(
            features, coords, n_neighbors=N_CHECK_NEIGHBOURS, metric="cosine")), 4),
        "neighbour_cluster_agreement_2d": round(
            neighbour_cluster_agreement(coords, labels, "euclidean"), 4),
        "neighbour_cluster_agreement_full": round(
            neighbour_cluster_agreement(features, labels, "cosine"), 4),
    }


def main_view(coords, factor=MAIN_VIEW_FACTOR):
    """True for postings in the main body of the map (see MAIN_VIEW_FACTOR)."""
    distance = np.linalg.norm(coords - np.median(coords, axis=0), axis=1)
    return distance <= factor * np.median(distance)


def build_table(ids, coords, clusters, postings):
    """One row per posting, in the order of the features file."""
    frame = pd.DataFrame({"id": ids, "x": coords[:, 0].round(4), "y": coords[:, 1].round(4),
                          "in_main_view": main_view(coords)})
    frame = frame.merge(clusters, on="id", how="left", validate="one_to_one")
    details = postings.set_index("id")[HOVER_COLUMNS]
    return frame.join(details, on="id")


# --- Drawing ---------------------------------------------------------------------------

# Fixed colour per cluster, so a cluster keeps its colour on every page and
# after filtering. The three role-type clusters get the three strongest hues;
# the five employer/ad-template clusters get the rest and are drawn fainter,
# so the eye goes to the role types first.
CLUSTER_COLOURS = {
    3: "#2a78d6",  # Machine learning & AI         (blue)
    5: "#eb6834",  # Data platform & pipelines     (orange)
    6: "#1baf7a",  # Business reporting & analysis (aqua)
    0: "#eda100",
    1: "#e87ba4",
    2: "#008300",
    4: "#6250d6",
    7: "#e34948",
}


def cluster_map(table, view_table=None):
    """Plotly scatter of the map, one trace per cluster. Used by the app too.

    view_table sets the opening view. The app passes the full table while
    drawing a filtered one, so the map does not jump around as filters change.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    order = list(ROLE_CLUSTERS) + [c for c in sorted(CLUSTER_COLOURS) if c not in ROLE_CLUSTERS]
    for cluster in order:
        group = table[table["cluster"] == cluster]
        if group.empty:
            continue
        is_role = cluster in ROLE_CLUSTERS
        salary = group["salary_mid"].map(lambda s: f"£{s:,.0f}" if pd.notna(s) else "not given")
        fig.add_trace(go.Scattergl(
            x=group["x"], y=group["y"], mode="markers",
            name=f"{CLUSTER_NAMES[cluster]} ({len(group)})",
            legendgroup="role" if is_role else "employer",
            legendgrouptitle_text="Role types" if is_role else "Employer / ad groups",
            marker=dict(size=7 if is_role else 6, color=CLUSTER_COLOURS[cluster],
                        opacity=0.85 if is_role else 0.45,
                        line=dict(width=0.5, color="white")),
            customdata=np.column_stack([group["title"], group["company"].fillna(""),
                                        group["title_family"], group["role_type"], salary,
                                        group["id"]]),
            hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                           "Title family: %{customdata[2]}<br>"
                           "Role type: %{customdata[3]}<br>"
                           "Salary (mid): %{customdata[4]}"
                           f"<extra>{CLUSTER_NAMES[cluster]}</extra>"),
        ))
    # The axes have no units (UMAP coordinates mean nothing on their own), so
    # they are hidden. The view opens on the main body; see MAIN_VIEW_FACTOR.
    view_table = table if view_table is None else view_table
    inside = view_table[view_table["in_main_view"]]
    pad_x = 0.05 * (inside["x"].max() - inside["x"].min())
    pad_y = 0.05 * (inside["y"].max() - inside["y"].min())
    fig.update_layout(
        title="Data job postings, placed by what the ad says (UMAP of MiniLM embeddings)",
        template="plotly_white", height=650,
        xaxis=dict(visible=False, range=[inside["x"].min() - pad_x, inside["x"].max() + pad_x]),
        yaxis=dict(visible=False, range=[inside["y"].min() - pad_y, inside["y"].max() + pad_y]),
        legend=dict(itemsizing="constant"), margin=dict(l=10, r=10, t=60, b=40),
    )
    outside = table[~table["in_main_view"]]
    if len(outside):
        top = outside["cluster_name"].value_counts()
        fig.add_annotation(
            text=(f"{len(outside)} postings sit outside this view, mostly "
                  f"'{top.index[0]}' ({top.iloc[0]}): near-identical ads from one "
                  "employer form their own island. Double-click to zoom out."),
            xref="paper", yref="paper", x=0, y=-0.05, showarrow=False,
            xanchor="left", font=dict(size=12, color="#52514e"))
    return fig


# --- Command line -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--postings", type=Path)
    parser.add_argument("--n-neighbors", type=int, default=UMAP_2D_SETTINGS["n_neighbors"])
    parser.add_argument("--min-dist", type=float, default=UMAP_2D_SETTINGS["min_dist"])
    parser.add_argument("--preview", action="store_true", help="also write an HTML map")
    args = parser.parse_args()

    path = args.postings or latest_postings()
    suffix = path.stem.replace("postings_", "", 1)
    out_dir = path.parent
    cluster_meta = json.loads((out_dir / f"clusters_{suffix}.meta.json").read_text())

    postings = pd.read_csv(path, dtype={"id": str})
    clusters = pd.read_csv(out_dir / f"clusters_{suffix}.csv", dtype={"id": str})
    ids, features = load_features(out_dir / cluster_meta["features"])
    labels = clusters.set_index("id").loc[ids, "cluster"].to_numpy()

    print(f"UMAP 2D on {cluster_meta['features']} "
          f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist})", flush=True)
    coords = project_2d(features, args.n_neighbors, args.min_dist)
    table = build_table(ids, coords, clusters, postings)
    scores = quality(features, coords, labels)

    table.to_csv(out_dir / f"projection_{suffix}.csv", index=False)
    meta = {"source_postings": path.name, "features": cluster_meta["features"],
            "clusters": f"clusters_{suffix}.csv", "seed": SEED,
            "umap": {**UMAP_2D_SETTINGS, "n_neighbors": args.n_neighbors,
                     "min_dist": args.min_dist},
            **scores}
    (out_dir / f"projection_{suffix}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(scores, indent=2))
    print(f"wrote projection_{suffix}.csv")

    if args.preview:
        out = out_dir / f"cluster_map_{suffix}.html"
        cluster_map(table).write_html(out, include_plotlyjs="cdn")
        print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
