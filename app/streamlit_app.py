"""Data Role Explorer: the Streamlit app (milestone 6).

Run from the repo root:
    streamlit run app/streamlit_app.py

This file only sets up the pages; each page lives in app/views/. Pages are
declared with st.navigation (rather than a pages/ folder) so their order and
titles are set in one place below.
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit puts app/ on the import path, not the repo root. Adding the root
# lets the pages reuse the pipeline's own code (src.match, src.projection, ...)
# instead of copying it, so the app and the analysis can never disagree.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load, snapshot_label, snapshot_suffix  # noqa: E402

st.set_page_config(page_title="Data Role Explorer", layout="wide")

suffix = snapshot_suffix()
load(suffix)  # warm the cache once, so the first page opens with data ready

views = Path(__file__).parent / "views"
page = st.navigation([
    st.Page(views / "overview.py", title="Market overview", default=True),
    st.Page(views / "cluster_map.py", title="Cluster map"),
    st.Page(views / "profiles.py", title="Cluster profiles"),
    st.Page(views / "match_me.py", title="Match me"),
])

with st.sidebar:
    st.caption(f"Data: {snapshot_label(suffix)}. UK data job ads from the official "
               "Adzuna API, cleaned and clustered offline.")
    st.caption("[Code and method on GitHub](https://github.com/dylanew/Data-Role-Explorer-)")

page.run()
