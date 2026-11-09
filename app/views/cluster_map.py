"""Cluster map: every ad as a dot, placed by what its text says (milestone 5 map)."""

import streamlit as st

from data import load, snapshot_suffix
from src.projection import cluster_map

data = load(snapshot_suffix())
table = data["projection"]
quality = data["projection_meta"]

st.title("Cluster map")
st.write(
    "Each dot is one job ad. Ads whose text means similar things sit close together. "
    "Colours are the 8 clusters found by K-means: three are **role types** (what the work "
    "is), five are **employer or ad groups** (ads that cluster because they share "
    "company text, not because the job is the same).")

# --- Filters, in one row above the chart ---------------------------------------------
left, right = st.columns(2)
families = sorted(table["title_family"].unique())
roles = sorted(table["role_type"].unique())
pick_families = left.multiselect("Job title", families, placeholder="All job titles")
pick_roles = right.multiselect("Role type", roles, placeholder="All role types",
                               help="Ads in an employer group get a role type from the "
                                    "skills they name, where they name enough.")
shown = table
if pick_families:
    shown = shown[shown["title_family"].isin(pick_families)]
if pick_roles:
    shown = shown[shown["role_type"].isin(pick_roles)]
st.caption(f"Showing {len(shown):,} of {len(table):,} ads. Hover over a dot for the ad; "
           "click a legend entry to hide a cluster; drag a box or lasso to list ads below.")

# The full table sets the view, so the map stays put while filters change.
fig = cluster_map(shown, view_table=table)
fig.update_layout(title=None, margin=dict(l=10, r=10, t=10, b=40))
event = st.plotly_chart(fig, on_select="rerun", selection_mode=("box", "lasso"),
                        key="cluster_map")

# --- Selected ads ----------------------------------------------------------------------
points = event.selection.points if event else []
if points:
    ids = [point["customdata"][5] for point in points if "customdata" in point]
    picked = shown[shown["id"].isin(ids)]
    st.subheader(f"{len(picked)} selected ads")
    st.dataframe(
        picked[["title", "company", "title_family", "cluster_name", "role_type",
                "seniority", "salary_mid"]],
        hide_index=True,
        column_config={"title_family": "Job title family", "cluster_name": "Cluster",
                       "role_type": "Role type",
                       "salary_mid": st.column_config.NumberColumn("Salary (mid)",
                                                                   format="£%d")})

with st.expander("How to read this map"):
    st.markdown(f"""
- The clusters were found in the full 384-dimensional sentence embeddings of each ad
  (MiniLM, with role names like "data engineer" blanked out so the clusters come from
  the work, not the title). UMAP only squeezes those embeddings into 2D to draw them;
  it never changes a cluster.
- Who sits next to whom is meaningful; the distance between far-apart groups is not.
- The axes have no units, so they are hidden.
- A few employers post near-identical ads, which form tight islands away from the
  rest. The map opens on the main body; double-click to zoom out.
- Map quality: trustworthiness {quality['trustworthiness']:.2f} (how often an ad's
  neighbours on the map were also its neighbours in 384-d; 1 is perfect), and
  {quality['neighbour_cluster_agreement_2d']:.0%} of an ad's 10 nearest dots share its
  cluster, against {quality['neighbour_cluster_agreement_full']:.0%} in 384-d, so the map
  does not make the clusters look more separate than they are.
""")
