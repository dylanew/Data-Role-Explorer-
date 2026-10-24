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
