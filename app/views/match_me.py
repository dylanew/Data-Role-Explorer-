"""Match me: paste your skills or CV, see your nearest role type and what's missing."""

import streamlit as st

from charts import hbar
from data import load, snapshot_suffix
from src.cluster import CLUSTER_NAMES
from src.match import MIN_SHARE, missing_skills, score_roles, skills_in_text
from src.projection import CLUSTER_COLOURS
from src.skills import SKILLS

data = load(snapshot_suffix())
profiles = data["role_profiles"]
all_skills = sorted(skill for group in SKILLS.values() for skill in group)

st.title("Match me")
st.write(
    "Paste a list of your skills or your CV. The app finds the skills in it with the same "
    "dictionary used on the job ads, then compares them with the skills each of the three "
    "role types asks for. Nothing you paste is stored or sent anywhere.")

text = st.text_area("Your skills or CV", height=180,
                    placeholder="e.g. Python, SQL, dbt, Airflow, built ETL pipelines on Azure "
                                "and Databricks, Power BI dashboards for stakeholders...")
found = skills_in_text(text)
mine = st.multiselect(
    "Skills found (add or remove any)", all_skills, default=found,
    key=f"skills_{hash(text)}",  # a new text resets the list to what was found in it
    help="Only skills in the project's dictionary of 99 can be matched.")

if not mine:
    st.info("Paste some text or pick skills above to see your match.")
    st.stop()

# --- Scores ---------------------------------------------------------------------
scores = score_roles(mine, profiles)
best = scores.iloc[0]
if best["coverage"] == 0:
    st.warning("None of your skills appear in the role types' profiles, so there is no "
               "match to show. The profiles below list what each role asks for.")
else:
    st.subheader(f"Closest role type: {best['cluster_name']}")
    st.write(f"You have {best['n_have']} of the {best['n_profile']} skills its ads ask for, "
             f"covering {best['coverage']:.0%} of its skill demand.")

st.plotly_chart(hbar(scores["cluster_name"], scores["coverage"],
                     "How much of each role's skill demand you cover",
                     value_format=".0%", hover_label="Coverage",
                     colours=[CLUSTER_COLOURS[c] for c in scores["cluster"]], height=230))

# --- Missing skills, one tab per role, best match first ---------------------------------
st.subheader("Skills to add")
tabs = st.tabs(list(scores["cluster_name"]))
for tab, cluster in zip(tabs, scores["cluster"]):
    with tab:
        gap = missing_skills(mine, profiles, cluster)
        if gap.empty:
            st.success("You have every skill in this role's profile.")
            continue
        fig = hbar(gap["skill"], gap["share"],
                   f"Skills {CLUSTER_NAMES[cluster]} ads ask for that you did not list",
                   value_format=".0%", hover_label="Share of ads",
                   colours=CLUSTER_COLOURS[cluster])
        fig.data[0].customdata = gap["lift"].values
        fig.data[0].hovertemplate = ("<b>%{y}</b><br>Share of ads: %{x:.0%}<br>"
                                     "%{customdata}x as common as across all ads"
                                     "<extra></extra>")
        st.plotly_chart(fig, key=f"gap_{cluster}")

with st.expander("How the match works"):
    st.markdown(f"""
1. **Skills in your text.** The same dictionary and rules as for the ads
   (`src/skills.py`): whole words, synonyms such as "sklearn" for scikit-learn.
2. **A profile per role type.** For each of the three role-type clusters: the share of
   its ads that name each skill. Skills named by fewer than {MIN_SHARE:.0%} of its ads are
   left out.
3. **Coverage.** Add up the shares of the profile skills you have, and divide by the
   total of all its shares. A skill that 20% of ads ask for counts ten times as much as
   one that 2% ask for.
4. **Skills to add** are the profile skills you didn't list, most asked-for first.

Only the three role types are matched. The other five clusters group ads by employer,
so matching to them would say nothing about your skills. Shares look small because
many ad snippets name no skill at all; the ranking still holds.
""")
