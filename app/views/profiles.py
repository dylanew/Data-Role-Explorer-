"""Cluster profiles: what each cluster is, and how clusters line up with job titles."""

import plotly.graph_objects as go
import streamlit as st

from charts import hbar
from data import cluster_order, load, snapshot_suffix
from src.cluster import CLUSTER_NAMES, ROLE_CLUSTERS
from src.projection import CLUSTER_COLOURS

data = load(snapshot_suffix())
postings, clusters, skills = data["postings"], data["clusters"], data["skills"]
profiles = data["profiles"].set_index("cluster")
merged = postings.merge(clusters[["id", "cluster"]], on="id")

st.title("Cluster profiles")
st.write(
    "K-means split the ads into 8 clusters using what each ad says about the work, with "
    "role names blanked out. Three clusters are clear **role types**. The other five "
    "group ads by **employer or ad template**, because the 500-character snippets Adzuna "
    "returns often hold little more than the company blurb.")

# --- Pick a cluster ------------------------------------------------------------------
def label(cluster):
    kind = "Role type" if cluster in ROLE_CLUSTERS else "Employer / ad group"
    return f"{CLUSTER_NAMES[cluster]}  ({kind})"


cluster = st.selectbox("Cluster", cluster_order(), format_func=label)
members = merged[merged["cluster"] == cluster]
profile = profiles.loc[cluster]
colour = CLUSTER_COLOURS[cluster]

cols = st.columns(4)
cols[0].metric("Ads", f"{len(members):,}", f"{len(members) / len(merged):.0%} of all",
               delta_color="off", delta_arrow="off")
salaries = members["salary_mid"].dropna()
cols[1].metric("Median advertised salary",
               f"£{salaries.median():,.0f}" if len(salaries) >= 10 else "Too few",
               f"from {len(salaries)} ads" if len(salaries) else None, delta_color="off", delta_arrow="off")
cols[2].metric("Most common job title", profile["main_title_family"].title(),
               f"{profile['main_title_family_share']:.0%} of the cluster", delta_color="off", delta_arrow="off")
senior = members["seniority"].isin(["senior", "lead", "principal", "staff", "head",
                                    "manager"]).mean()
cols[3].metric("Senior or above in title", f"{senior:.0%}")

left, right = st.columns(2)
with left:
    hits = skills[skills["in_description"] & skills["id"].isin(members["id"])]
    overall = (skills[skills["in_description"]].groupby("skill")["id"].nunique()
               / len(merged))
    share = (hits.groupby("skill")["id"].nunique() / len(members)) \
        .sort_values(ascending=False).head(12)
    if len(share):
        fig = hbar(share.index, share.values, "Skills its ads ask for (share of ads)",
                   value_format=".0%", hover_label="Share of ads", colours=colour)
        lift = (share / overall[share.index]).round(1)
        fig.data[0].customdata = lift.values
        fig.data[0].hovertemplate = ("<b>%{y}</b><br>Share of ads: %{x:.0%}<br>"
                                     "%{customdata}x as common as across all ads<extra></extra>")
        st.plotly_chart(fig)
    else:
        st.info("No ad in this cluster names a skill from the dictionary.")
    st.caption(f"Words that set this cluster's ads apart: {profile['top_terms']}.")

with right:
    families = members["title_family"].value_counts(normalize=True)
    st.plotly_chart(hbar(families.index, families.values,
                         "What the ads call the job (share of ads)", value_format=".0%",
                         hover_label="Share of ads", colours=colour))

with st.expander(f"Browse the {len(members):,} ads in this cluster"):
    st.dataframe(
        members[["title", "company", "location", "seniority", "salary_mid", "redirect_url"]],
        hide_index=True,
        column_config={"salary_mid": st.column_config.NumberColumn("Salary (mid)",
                                                                   format="£%d"),
                       "redirect_url": st.column_config.LinkColumn("Ad", display_text="Open")})

# --- Clusters vs job titles: the research question -------------------------------------
st.header("Do the clusters line up with job titles?")
st.write(
    "Each column is one job title family; the cells say where its ads ended up, so each "
    "column adds to 100%. If clusters simply repeated job titles, every column would have "
    "one dark cell.")
order = cluster_order()
table = (merged.groupby(["cluster", "title_family"]).size().unstack(fill_value=0)
         .reindex(order))
table = table[table.sum().sort_values(ascending=False).index]
shares = table / table.sum()
fig = go.Figure(go.Heatmap(
    z=shares.values,
    x=[f"{f}<br>({n})" for f, n in table.sum().items()],
    y=[CLUSTER_NAMES[c] for c in order],
    colorscale=[[0, "#f4f7fb"], [1, "#1c4f8f"]], zmin=0, zmax=shares.values.max(),
    text=[[f"{v:.0%}" for v in row] for row in shares.values], texttemplate="%{text}",
    customdata=table.values,
    hovertemplate="%{y}<br>%{x}<br>%{z:.0%} of this title (%{customdata} ads)<extra></extra>",
    xgap=2, ygap=2, colorbar=dict(title="Share", tickformat=".0%"),
))
fig.update_layout(template="plotly_white", height=460, margin=dict(l=10, r=10, t=10, b=10),
                  yaxis=dict(autorange="reversed", automargin=True),
                  xaxis=dict(side="top"))
st.plotly_chart(fig)
st.caption(
    f"Adjusted Rand index between clusters and title families: "
    f"{data['cluster_meta']['ari_vs_title']:.2f} (0 = unrelated, 1 = identical). Data "
    "engineers and BI analysts mostly land in one role type each; data scientists and ML "
    "engineers spread across machine learning and several employer groups. 'Analytics "
    "engineer' has only 4 ads.")
