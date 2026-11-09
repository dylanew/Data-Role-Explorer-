"""Draw the README's job title vs role type heatmap.

Run from the repo root:  python docs/make_readme_charts.py
Reads data/processed/clusters_gb_2026-10-24.csv, writes
docs/title_vs_role_type_gb_2026-10-24.png. Needs matplotlib (not used by
the pipeline or the app, so it is not in requirements.txt).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SNAPSHOT = "gb_2026-10-24"
ROOT = Path(__file__).resolve().parents[1]
CLUSTERS = ROOT / "data" / "processed" / f"clusters_{SNAPSHOT}.csv"
OUT = ROOT / "docs" / f"title_vs_role_type_{SNAPSHOT}.png"

# The three work-based clusters keep their own column; the five
# employer / ad-template clusters are summed into one.
ROLE_COLUMNS = {
    "Machine learning & AI": "ML & AI",
    "Data platform & pipelines": "Data platform\n& pipelines",
    "Business reporting & analysis": "Reporting\n& analysis",
}
EMPLOYER_COLUMN = "Employer / ad\ngroups (5)"

# Titles with at least 60 ads, largest first.
TITLES = {
    "data scientist": "Data scientist",
    "data engineer": "Data engineer",
    "data analyst": "Data analyst",
    "machine learning engineer": "ML engineer",
    "ai engineer": "AI engineer",
    "bi analyst": "BI analyst",
}

# One-hue sequential blue ramp, light (near 0%) to dark.
BLUES = ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def shares():
    df = pd.read_csv(CLUSTERS)
    df = df[df["title_family"].isin(TITLES)]
    df["column"] = df["cluster_name"].map(ROLE_COLUMNS).fillna(EMPLOYER_COLUMN)
    table = pd.crosstab(df["title_family"], df["column"], normalize="index")
    counts = df["title_family"].value_counts()
    table = table.loc[list(TITLES), list(ROLE_COLUMNS.values()) + [EMPLOYER_COLUMN]]
    return table, counts


def main():
    table, counts = shares()
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    ax.imshow(table.values, cmap=cmap, vmin=0, vmax=0.6, aspect="auto")

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.values[i, j]
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if value >= 0.35 else "#1f1f1f")

    ax.set_xticks(range(table.shape[1]), table.columns, fontsize=9)
    ax.set_yticks(range(table.shape[0]),
                  [f"{TITLES[t]}  ({counts[t]})" for t in table.index], fontsize=9)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 2px white gaps between cells
    ax.set_xticks([x - 0.5 for x in range(1, table.shape[1])], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, table.shape[0])], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    ax.set_title("Where each job title's ads ended up (share of the title's ads)",
                 fontsize=11, loc="left", pad=40)
    fig.text(0.01, 0.01, "UK Adzuna ads, 2026-10-24. Clusters built with role names "
             "masked. Rows sum to 100%.", fontsize=8, color="#5f5f5f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
