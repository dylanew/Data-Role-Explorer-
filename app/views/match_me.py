"""Match me: paste or upload your CV, see your closest role type and what's missing."""

import streamlit as st

from charts import hbar
from data import load, load_centres, model, snapshot_suffix
from src.cluster import CLUSTER_NAMES
from src.match import (CLOSE_CALL, MIN_SHARE, closest_roles, embed_text, is_close_call,
                       missing_skills, pdf_text, score_roles, skills_in_text)
from src.projection import CLUSTER_COLOURS
from src.skills import SKILLS

suffix = snapshot_suffix()
profiles = load(suffix)["role_profiles"]
centres = load_centres(suffix)
all_skills = sorted(skill for group in SKILLS.values() for skill in group)

st.title("Match me")
st.write(
    "Paste your skills or CV, or upload your CV as a PDF. The app reads it with the same "
    "language model used to group the job ads, finds the closest of the three role types, "
    "then lists the skills that role's ads ask for that you don't mention. Nothing you "
    "paste or upload is stored.")

# --- Input: pasted text or a PDF ----------------------------------------------------
source = st.segmented_control("Input", ["Paste text", "Upload a PDF"], default="Paste text",
                              label_visibility="collapsed")
text = ""
if source == "Upload a PDF":
    upload = st.file_uploader("Your CV (PDF)", type="pdf")
    if upload is not None:
        try:
            text = pdf_text(upload)
        except Exception:  # a damaged or encrypted file: say so instead of crashing
            st.error("Could not read that PDF. Try another file, or paste the text instead.")
        else:
            if not text:
                st.warning("No text found in that PDF. It may be a scanned image; "
                           "paste the text instead.")
            else:
                with st.expander(f"Text read from {upload.name} ({len(text):,} characters)"):
                    st.text(text)
else:
    text = st.text_area("Your skills or CV", height=180,
                        placeholder="e.g. Python, SQL, dbt, Airflow, built ETL pipelines on "
                                    "Azure and Databricks, Power BI dashboards for "
                                    "stakeholders...")

if not text.strip():
    st.info("Paste some text or upload a PDF to see your match.")
    st.stop()

# --- Step 1: the closest role, from embeddings -------------------------------------------
with st.spinner("Reading your text..."):
    vector = embed_text(text, model())
if vector is None:
    st.info("There is nothing left to match once job titles are removed. "
            "Add a few lines about what you do or the tools you use.")
    st.stop()
roles = closest_roles(vector, centres, CLUSTER_NAMES)
best, second = roles.iloc[0], roles.iloc[1]

if is_close_call(roles):
    st.subheader(f"Close call: {best['cluster_name']} or {second['cluster_name']}")
    st.write("Your text is about as close to both. The skills below can help you decide "
             "which way to lean.")
else:
    st.subheader(f"Closest role type: {best['cluster_name']}")
fig = hbar(roles["cluster_name"], roles["similarity"],
           "How close your text is to each role type (cosine similarity)",
           value_format=".2f", hover_label="Similarity",
           colours=[CLUSTER_COLOURS[c] for c in roles["cluster"]], height=230)
st.plotly_chart(fig)
st.caption("1 would be identical in meaning, 0 unrelated. A CV never scores as high as "
           "the ads themselves, because it is written differently; what matters is how the "
           "three roles compare.")

# --- Step 2: skills, from the dictionary -------------------------------------------------
found = skills_in_text(text)
mine = st.multiselect(
    "Skills found in your text (add or remove any)", all_skills, default=found,
    key=f"skills_{hash(text)}",  # a new text resets the list to what was found in it
    help="Only skills in the project's dictionary of 99 can be found. Changing this list "
         "updates the skills to add, not the closest role, which comes from your whole text.")

st.subheader("Skills to add")
coverage = score_roles(mine, profiles).set_index("cluster")
tabs = st.tabs(list(roles["cluster_name"]))
for tab, cluster in zip(tabs, roles["cluster"]):
    with tab:
        cov = coverage.loc[cluster]
        st.caption(f"You list {cov['n_have']} of the {cov['n_profile']} skills in this "
                   f"role's profile, covering {cov['coverage']:.0%} of its skill demand.")
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
**Closest role (the language model).**
1. Job titles such as "data engineer" are blanked out of your text, as they were in the
   ads, so the match comes from what you do rather than what you were called.
2. Your text is cut into pieces about the length of an ad snippet, each piece is turned
   into a vector by the MiniLM sentence model, and the pieces are averaged.
3. Each role type's centre is the average vector of its ads. The closest centre (cosine
   similarity) is your match. If the top two are within {CLOSE_CALL}, it is called a
   close call.

**Skills to add (the skill dictionary).**
1. Skills are found in your text with the same dictionary and rules as the ads
   (`src/skills.py`): whole words, synonyms such as "sklearn" for scikit-learn.
2. Each role type's profile is the share of its ads that name each skill (skills named
   by fewer than {MIN_SHARE:.0%} of its ads are left out).
3. Skills to add are the profile skills you didn't list, most asked-for first.

Only the three role types are matched. The other five clusters group ads by employer,
so matching to them would say nothing about your work. Shares look small because many
ad snippets name no skill at all; the order still holds.
""")
