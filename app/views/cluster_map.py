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
    "Every ad carries three labels, which you can filter or colour the dots by:")
st.markdown(
    "- **Cluster**: which of the 8 K-means groups its text fell in. Three are role types "
    "(what the work is); five are employer or ad groups (ads that cluster because they "
    "share company text, not because the job is the same).\n"
    "- **Role type**: the kind of work. Ads in a role-type cluster take that cluster's "
    "name; ads in an employer group get one from the skills they name, if they name "
    "enough; the rest are *Not stated in the ad*.\n"
    "- **Job title in the ad**: what the employer calls the job, e.g. \"data engineer\".")

# --- Controls, in one row above the chart ---------------------------------------------
left, middle, right = st.columns([2, 2, 1.4])
families = sorted(table["title_family"].unique())
roles = sorted(table["role_type"].unique())
pick_families = left.multiselect("Job title in the ad", families,
                                 placeholder="All job titles")
pick_roles = middle.multiselect("Role type (from cluster or skills)", roles,
                                placeholder="All role types")
colour_by = right.radio("Colour dots by", ["Cluster", "Role type"], horizontal=True,
                        help="Cluster: where the ad's text fell. Role type: the kind of "
                             "work, from the cluster or from the skills the ad names.")
shown = table
if pick_families:
    shown = shown[shown["title_family"].isin(pick_families)]
if pick_roles:
    shown = shown[shown["role_type"].isin(pick_roles)]

# Say where the shown ads' role types came from, so mixed colours make sense.
source = shown["role_type_source"].value_counts()
st.caption(
    f"Showing {len(shown):,} of {len(table):,} ads: {source.get('cluster', 0):,} in a "
    f"role-type cluster, {source.get('skills', 0):,} in an employer group but given a role "
    f"type from their skills, {source.get('none', 0):,} with no role type. Hover over a "
    "dot for the ad; click a legend entry to hide it; drag a box or lasso to list ads below.")

# The full table sets the view, so the map stays put while filters change.
fig = cluster_map(shown, view_table=table, colour_by=colour_by.lower().replace(" ", "_"))
fig.update_layout(title=None, margin=dict(l=10, r=10, t=40, b=40))  # room for the toolbar
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
        column_config={"title_family": "Job title in the ad", "cluster_name": "Cluster",
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
