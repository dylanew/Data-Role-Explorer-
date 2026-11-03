# Data-Role-Explorer-
The "get me a job" project.

## How to run: data collection (milestone 1)

1. Get free API keys at https://developer.adzuna.com/ (sign up, then copy the
   App ID and App Key from your dashboard).
2. Set up the project:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env   # then paste your keys into .env
   ```
3. Collect postings:
   ```bash
   python -m src.collect                 # UK, 10 pages per search term
   python -m src.collect --country us --pages 5
   ```
   This writes `data/raw/adzuna_<country>_<date>.jsonl` (one posting per line)
   and a matching `.meta.json` describing the run.
4. Run the tests (no keys needed): `python -m pytest`

## How to run: cleaning (milestone 2)

```bash
python -m src.clean                                      # newest raw snapshot
python -m src.clean data/raw/adzuna_gb_2026-10-24.jsonl  # a specific one
```

This writes `data/processed/postings_<country>_<date>.csv` (one row per job)
and a `.meta.json` with how many rows each step removed. The UK snapshot of
2026-10-24 goes from 2,454 raw rows to 2,262 unique ids to 1,995 jobs, of
which 1,857 are in scope (data roles that aren't training-course ads).

Cleaning rules (all defined at the top of `src/clean.py`):

| Step | Rule | Why |
|---|---|---|
| Same id | One row per Adzuna id; `search_terms` lists every term that found it | A job can match several searches |
| Reposts | Same title + company + description = one job; `n_locations` counts the places it was posted | Some advertisers post one ad in many towns |
| Not on description alone | Different titles at one company are kept apart | Descriptions are 500-char snippets that often start with company boilerplate |
| Text | HTML tags, entities and zero-width characters removed | Clean input for skill matching |
| `title_clean` | Lowercase; ML/BI/Sr written out; brackets, " - extra text", work-mode words and codes removed | Make titles comparable |
| `seniority` | First matching keyword: intern, entry, junior, head, manager, principal, staff, lead, senior, else unspecified | Separates level from role |
| `title_family` | First matching rule: analytics engineer, ML engineer, AI engineer, data engineer, data scientist, BI analyst, data analyst, else other | The "official title" the clusters are compared against |
| Salary | Advertised salaries go in `salary_min/max/mid`; Adzuna's own predictions (about 60%) go in `salary_estimate`; advertised values under £10k (day or hourly rates) are dropped; nothing is imputed | Keep real figures separate from a model's guesses |
| Flags | `is_data_role` (title names a data role), `is_training_ad` (trainee / placement-programme course ads) | Rows are flagged, not deleted, so later steps choose |

## How to run: skill extraction (milestone 3)

```bash
python -m src.skills                                          # newest cleaned file
python -m src.skills data/processed/postings_gb_2026-10-24.csv
```

This writes three files to `data/processed/`:

- `skills_<country>_<date>.csv`: one row per posting and skill found, with
  `in_title`, `in_description` and the exact `matched_text`, so every match
  can be checked by eye.
- `skill_frequency_<country>_<date>.csv`: one row per skill: how many in-scope
  postings mention it, the share overall, and the share within each title family.
- `skills_<country>_<date>.meta.json`: coverage numbers, including how many
  postings have no skills at all.

How matching works (dictionary and rules at the top of `src/skills.py`):

| Rule | Example | Why |
|---|---|---|
| Curated dictionary of 99 skills in 10 categories, each with synonyms | `scikit-learn`: "scikit-learn", "sklearn" | Transparent and easy to change, unlike a learned extractor |
| Whole words only | "scala" does not match "scalable" | Avoid matches inside other words |
| Spaces and hyphens are interchangeable | "power bi" matches "Power-BI" and "PowerBI" | Ads write tool names inconsistently |
| Case ignored, except short words that are also ordinary English | "Excel" but not "excel in this role"; "SAS", "SAP", "Fabric" | Fewer false matches |
| Special rules for "R" and "RAG" | "Python or R" counts; "R&D", "A&R", "R-2079" do not; "RAG status" (red/amber/green) does not | A single letter needs its own rule |
| "Go" only as "Golang" | "ready to go live" does not count | "Go" is nearly always the verb |
| Frequencies count descriptions only | "Machine Learning Engineer" in a title is not the skill "machine learning" | Titles are what clusters get compared with later, so counting them would be circular |

**Sparsity.** Adzuna descriptions are 500-character snippets that often open
with company boilerplate, so many postings mention no skill. In the UK
snapshot of 2026-10-24, 1,070 of the 1,857 in-scope postings (58%) have no
skill in their description (947 if titles are also searched). Postings that
do mention skills have between 1 and 15.

## How to run: clustering (milestone 4, in progress)

```bash
python -m src.embed                 # features for every text version (TF-IDF + sentence embeddings)
python -m src.cluster compare       # score K-means and HDBSCAN on every feature set
python -m src.cluster final features_<...>.npz --kmeans 8   # once a setting is chosen
```

The clusters are built from the description with **role names masked and
employer boilerplate removed** (`--text masked`), so they come from what the
ad says about the work rather than from the job title they will be compared
with. The unmasked versions are kept for comparison. Sentence embeddings need
the model downloaded from huggingface.co; the chosen model, metric and number
of clusters are still to be decided.
