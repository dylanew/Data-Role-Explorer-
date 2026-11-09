"""Market overview: what the job ads in the snapshot look like."""

import plotly.graph_objects as go
import streamlit as st

from charts import hbar
from data import NEUTRAL, load, snapshot_label, snapshot_suffix

suffix = snapshot_suffix()
data = load(suffix)
postings, skills = data["postings"], data["skills"]

st.title("The UK data jobs market")
st.write(
    f"{len(postings):,} data job ads from the {snapshot_label(suffix)}, after removing "
    "duplicates, reposts and training-course ads. What do employers ask for, and what "
    "do they pay?")

# --- Headline numbers ---------------------------------------------------------
advertised = postings["salary_source"].eq("advertised")
with_skill = skills.loc[skills["in_description"], "id"].nunique()
cols = st.columns(4)
cols[0].metric("Job ads", f"{len(postings):,}")
cols[1].metric("Employers", f"{postings['company'].nunique():,}")
cols[2].metric("Advertise a salary", f"{advertised.mean():.0%}",
               help="Salaries Adzuna predicted itself are left out everywhere in this app.")
cols[3].metric("Name at least one skill", f"{with_skill / len(postings):.0%}",
               help="Adzuna only returns a 500-character snippet of each ad, which often "
                    "opens with company text, so many snippets name no skill.")

# --- Filter ----------------------------------------------------------------------
families = postings["title_family"].value_counts()
choice = st.selectbox("Job title", ["All data jobs"] + list(families.index),
                      help="The title family each ad was put in from its job title.")
if choice != "All data jobs":
    postings = postings[postings["title_family"] == choice]
    skills = skills[skills["id"].isin(postings["id"])]
    advertised = postings["salary_source"].eq("advertised")

left, right = st.columns(2)

# --- Skills ----------------------------------------------------------------------
with left:
    counts = (skills[skills["in_description"]].groupby("skill")["id"].nunique()
              .sort_values(ascending=False).head(15))
    st.plotly_chart(hbar(counts.index, counts / len(postings),
                         "Most-asked-for skills (share of ads)", value_format=".0%",
                         hover_label="Share of ads"))
    st.caption("Counted from the ad text only, not the job title, using the skill "
               "dictionary in src/skills.py.")

# --- Title families (only for "all") or seniority -----------------------------------
with right:
    if choice == "All data jobs":
        st.plotly_chart(hbar(families.index, families.values, "Ads by job title"))
    seniority = postings["seniority"].value_counts()
    st.plotly_chart(hbar(seniority.index, seniority.values, "Ads by seniority in the title"))

# --- Salary ----------------------------------------------------------------------
st.subheader("Advertised salaries")
paid = postings[advertised]
by_family = paid.groupby("title_family")["salary_mid"]
shown = by_family.count()[lambda n: n >= 10].index  # a box of 3 salaries says little
if len(shown):
    order = by_family.median()[shown].sort_values(ascending=False).index
    fig = go.Figure()
    for family in order:
        values = paid.loc[paid["title_family"] == family, "salary_mid"]
        fig.add_trace(go.Box(
            x=values, name=f"{family} ({len(values)})", orientation="h",
            marker=dict(color=NEUTRAL, size=4), line=dict(width=1.5), boxpoints="outliers",
            hovertemplate="£%{x:,.0f}<extra></extra>"))
    fig.update_layout(template="plotly_white", showlegend=False,
                      height=max(260, 45 * len(order) + 80),
                      xaxis=dict(title="Annual salary, middle of the advertised range",
                                 tickprefix="£", tickformat=",.0f", gridcolor="#eeeeee"),
                      yaxis=dict(autorange="reversed", automargin=True),
                      margin=dict(l=10, r=20, t=20, b=40))
    st.plotly_chart(fig)
st.caption(f"{advertised.sum():,} of {len(postings):,} ads advertise a salary. Day and hourly "
           "rates are left out, and job titles with fewer than 10 salaries are not drawn. "
           "The box spans the middle half of salaries; the line inside is the median.")

# --- Where ---------------------------------------------------------------------
regions = postings["region"].fillna("Not given").value_counts().head(10)
st.plotly_chart(hbar(regions.index, regions.values, "Ads by region (top 10)"))
