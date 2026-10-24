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
