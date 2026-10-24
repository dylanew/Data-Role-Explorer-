"""Clean a raw Adzuna snapshot into one tidy table of postings.

Usage (from the repo root):
    python -m src.clean                                   # newest snapshot
    python -m src.clean data/raw/adzuna_gb_2026-10-24.jsonl

What it does, in order:
    1. Load the raw JSON Lines file and keep only the fields we use.
    2. Clean the text: strip HTML tags and entities from titles and
       descriptions, collapse whitespace.
    3. Deduplicate in two passes:
         a. Same Adzuna id: the posting was found by more than one search
            term. Keep one row and remember every term that found it.
         b. Same title + company + description: one ad posted in several
            towns. Keep one row and count how many places it was posted.
    4. Normalise titles: a cleaned title, a seniority level, and a broad
       "title family" (data scientist, data engineer, ...).
    5. Handle salary: separate advertised salaries from Adzuna's estimates,
       and drop values that are clearly day or hourly rates.
    6. Flag postings that are probably not data roles, and training-course
       ads, so later steps can leave them out.
    7. Write data/processed/postings_<country>_<date>.csv plus a
       .meta.json that records how many rows each step removed.

Design notes:
    - Nothing is thrown away silently. Rows we think are off-topic are
      flagged, not deleted, and the raw title is kept next to the cleaned one.
    - We do not deduplicate on description alone. Adzuna only gives us the
      first 500 characters, which often is company boilerplate ("Why
      Faculty? ..."), so different jobs at one company can share it.
    - Every rule is a plain regular expression or keyword list near the top
      of this file, so it is easy to read, test and change.
"""

import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Advertised salaries below this are almost always day or hourly rates
# (e.g. "£500" for a contract day rate), not annual pay.
MIN_ANNUAL_SALARY = 10_000

# --- Title rules -----------------------------------------------------------

# Short forms written out, so "ML Engineer" and "Machine Learning Engineer"
# end up the same. "power bi" is protected because it is a tool name.
ABBREVIATIONS = [
    (r"\bml\b", "machine learning"),
    (r"(?<!power )\bbi\b", "business intelligence"),
    (r"\b(sr|snr)\b\.?", "senior"),
    (r"\b(jr|jnr)\b\.?", "junior"),
    (r"\bmgr\b", "manager"),
]

# Words that describe the contract or place, not the role.
TITLE_NOISE = (
    r"\b(remote|hybrid|on-?site|contract|permanent|fixed[- ]term|ftc|"
    r"full[- ]time|part[- ]time|inside ir35|outside ir35|ir35|uk|london|"
    r"no experience needed|graduate considered)\b"
)

# Checked in order against the raw title; the first match wins. More
# specific levels come first, so "Senior / Lead" counts as lead.
SENIORITY_RULES = [
    ("intern", r"\b(intern|internship|summer analyst)\b"),
    ("entry", r"\b(graduate|grad|trainee|apprentice|apprenticeship|entry[- ]level)\b"),
    ("junior", r"\b(junior|jr|jnr)\b"),
    ("head", r"\b(head of|director|vp|vice president|chief)\b"),
    ("manager", r"\bmanager\b"),
    ("principal", r"\bprincipal\b"),
    ("staff", r"\bstaff\b"),
    ("lead", r"\b(lead|leader|team lead)\b"),
    ("senior", r"\b(senior|sr|snr)\b"),
]
DEFAULT_SENIORITY = "unspecified"

# Checked in order against the cleaned title; the first match wins. More
# specific families come first, so "analytics engineer" is not caught by
# the data engineer rule and "ML engineer" not by the data scientist rule.
FAMILY_RULES = [
    ("analytics engineer", r"\banalytics engineer"),
    ("machine learning engineer",
     r"\bmachine learning (?:\w+ )?(?:engineer|engineering|developer|ops)\b|\bmlops\b"),
    ("ai engineer",
     r"\b(ai|artificial intelligence|genai|llm)\b.*\b(engineer|engineering|developer)\b"),
    ("data engineer",
     r"\bdata (engineer|engineering)\b|\b(etl|data platform|big data) (engineer|developer)\b"),
    ("data scientist",
     r"\bdata scien|\b(applied|research|machine learning) scientist\b|"
     r"\bmachine learning researcher\b"),
    ("bi analyst",
     r"\bbusiness intelligence\b|\bpower bi\b|\b(bi|reporting) developer\b"),
    ("data analyst",
     r"\bdata analy|\b(insights?|analytics|reporting|product|marketing|"
     r"commercial|pricing|quantitative|statistical) analyst\b"),
]
DEFAULT_FAMILY = "other"

# A title mentioning any of these is treated as a data role. Titles without
# them (e.g. "Software Engineer", "Product Manager") were found only because
# the search phrase appeared in their description.
DATA_ROLE_WORDS = (
    r"\b(?:data|analyst|analytics|analysis|insights?|machine learning|ml|mlops|"
    r"ai|artificial intelligence|bi|business intelligence|scientist|science|"
    r"statistic\w*|quantitative|reporting)\b"
)

# Training providers advertise paid courses as jobs ("Data Analyst Placement
# Programme - No Experience Needed"). They are not real vacancies.
TRAINING_AD = r"\b(?:placement programme|no experience needed|trainee)\b"


# --- Text helpers ------------------------------------------------------------

def clean_text(text):
    """Remove HTML tags and entities, and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    # Adzuna sometimes writes newlines as the broken entity "&xa;".
    text = text.replace("&xa;", " ")
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[​-‍﻿]", "", text)   # invisible zero-width characters
    text = re.sub(r"\s+", " ", text)
    # Snippets end with "…" where Adzuna cut them off.
    return text.strip().rstrip("…").strip()


def normalise_title(title):
    """Turn a raw job title into a short, lowercase, comparable one.

    "Senior ML Engineer - Fulfillment (Hybrid)" -> "senior machine learning engineer"
    """
    t = title.lower().replace("_", " ")
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)        # bracketed extras
    # Keep the part before " - ", ",", ":" or " | ", which is usually the
    # role ("Data Scientist - Water Sector"). If that part names no data
    # role ("Senior Manager, Data Science"), keep the next part as well.
    parts = re.split(r" [-–|] |,|:", t)
    t = parts[0]
    if len(parts) > 1 and not re.search(DATA_ROLE_WORDS, t):
        t = t + " " + parts[1]
    for pattern, replacement in ABBREVIATIONS:
        t = re.sub(pattern, replacement, t)
    t = re.sub(TITLE_NOISE, " ", t)
    t = re.sub(r"\b\w*\d\w*\b", " ", t)           # reference codes, years
    t = re.sub(r"\b(i|ii|iii|iv)\s*$", " ", t)    # levels like "Data Scientist III"
    t = re.sub(r"[^\w&/+ ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" /&+")
    return t


def first_match(text, rules, default):
    """Return the label of the first (label, regex) rule that matches."""
    for label, pattern in rules:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return default


# --- Pipeline steps ----------------------------------------------------------

def load_raw(path):
    """Read a raw snapshot and flatten the fields we keep into a DataFrame."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            area = p.get("location", {}).get("area", [])
            rows.append({
                "id": str(p["id"]),
                "title_raw": p.get("title", ""),
                "company": p.get("company", {}).get("display_name"),
                "location": p.get("location", {}).get("display_name"),
                "region": area[1] if len(area) > 1 else None,
                "latitude": p.get("latitude"),
                "longitude": p.get("longitude"),
                "category": p.get("category", {}).get("label"),
                "contract_type": p.get("contract_type"),
                "contract_time": p.get("contract_time"),
                "created": p.get("created"),
                "description_raw": p.get("description", ""),
                "salary_min_raw": p.get("salary_min"),
                "salary_max_raw": p.get("salary_max"),
                "salary_is_predicted": p.get("salary_is_predicted") == "1",
                "search_term": p.get("search_term"),
                "redirect_url": p.get("redirect_url"),
            })
    df = pd.DataFrame(rows)
    df["created"] = pd.to_datetime(df["created"], utc=True)
    return df


def merge_same_id(df):
    """Pass 1: one row per Adzuna id, keeping every search term that found it."""
    terms = (df.groupby("id")["search_term"]
               .agg(lambda s: ";".join(sorted(set(s))))
               .rename("search_terms"))
    out = (df.drop(columns="search_term")
             .drop_duplicates("id")
             .merge(terms, on="id"))
    out["n_search_terms"] = out["search_terms"].str.count(";") + 1
    return out


def merge_reposts(df):
    """Pass 2: one row per ad posted in several places.

    Postings with the same title, company and description are treated as one
    job. We keep the earliest-created copy (ties broken by id, so the result
    is the same every run), combine their search terms, and count locations.
    """
    df = df.assign(_key=df["title"].str.lower() + " | "
                   + df["company"].fillna("") + " | " + df["description"])

    grouped = df.groupby("_key")
    n_locations = grouped["location"].nunique().rename("n_locations")
    terms = (grouped["search_terms"]
               .agg(lambda s: ";".join(sorted(set(";".join(s).split(";")))))
               .rename("search_terms"))

    out = (df.sort_values(["created", "id"])
             .drop_duplicates("_key")
             .drop(columns="search_terms")
             .join(terms, on="_key")
             .join(n_locations, on="_key")
             .drop(columns="_key"))
    out["n_search_terms"] = out["search_terms"].str.count(";") + 1
    return out


def handle_salary(df):
    """Split salaries into advertised figures and Adzuna's estimates.

    About 60% of Adzuna salaries are its own predictions (a single number),
    not what the employer advertised. Mixing them would make salary charts
    reflect Adzuna's model, so they go in their own column. Advertised
    salaries under MIN_ANNUAL_SALARY are day or hourly rates and are set to
    missing. Missing salaries are left missing, not filled in.
    """
    s_min = df["salary_min_raw"]
    s_max = df["salary_max_raw"].fillna(s_min)
    advertised = ~df["salary_is_predicted"] & s_min.notna()
    is_rate = advertised & (s_min < MIN_ANNUAL_SALARY)
    keep = advertised & ~is_rate

    df["salary_min"] = s_min.where(keep)
    df["salary_max"] = s_max.where(keep)
    df["salary_mid"] = (df["salary_min"] + df["salary_max"]) / 2
    df["salary_estimate"] = s_min.where(df["salary_is_predicted"])

    df["salary_source"] = "missing"
    df.loc[df["salary_is_predicted"] & s_min.notna(), "salary_source"] = "adzuna_estimate"
    df.loc[keep, "salary_source"] = "advertised"
    df.loc[is_rate, "salary_source"] = "dropped_rate"
    return df.drop(columns=["salary_min_raw", "salary_max_raw", "salary_is_predicted"])


def clean(df):
    """Run every cleaning step and return (clean DataFrame, step counts)."""
    counts = {"raw_rows": len(df)}

    df["title"] = df["title_raw"].map(clean_text)
    df["description"] = df["description_raw"].map(clean_text)
    df = df.drop(columns="description_raw")

    df = merge_same_id(df)
    counts["after_merging_same_id"] = len(df)
    df = merge_reposts(df)
    counts["after_merging_reposts"] = len(df)

    df["title_clean"] = df["title"].map(normalise_title)
    df["seniority"] = df["title"].map(
        lambda t: first_match(t, SENIORITY_RULES, DEFAULT_SENIORITY))
    df["title_family"] = df["title_clean"].map(
        lambda t: first_match(t, FAMILY_RULES, DEFAULT_FAMILY))

    df = handle_salary(df)

    df["is_data_role"] = df["title"].str.contains(DATA_ROLE_WORDS, case=False, regex=True)
    df["is_training_ad"] = df["title"].str.contains(TRAINING_AD, case=False, regex=True)

    columns = [
        "id", "title", "title_clean", "seniority", "title_family",
        "is_data_role", "is_training_ad", "company", "location", "region",
        "n_locations", "latitude", "longitude", "category", "contract_type",
        "contract_time", "created", "salary_min", "salary_max", "salary_mid",
        "salary_estimate", "salary_source", "search_terms", "n_search_terms",
        "description", "redirect_url",
    ]
    df = df.drop(columns="title_raw")[columns].sort_values("id").reset_index(drop=True)
    return df, counts


def summarise(df, counts):
    """Counts saved next to the output, so each run is easy to check."""
    in_scope = df["is_data_role"] & ~df["is_training_ad"]
    return {
        "row_counts": counts,
        "rows_out": len(df),
        "rows_in_scope": int(in_scope.sum()),
        "not_data_role": int((~df["is_data_role"]).sum()),
        "training_ads": int(df["is_training_ad"].sum()),
        "title_family": df["title_family"].value_counts().to_dict(),
        "seniority": df["seniority"].value_counts().to_dict(),
        "salary_source": df["salary_source"].value_counts().to_dict(),
    }


def latest_snapshot(raw_dir=RAW_DIR):
    snapshots = sorted(Path(raw_dir).glob("adzuna_*.jsonl"))
    if not snapshots:
        raise SystemExit(f"No snapshots in {raw_dir}. Run python -m src.collect first.")
    return snapshots[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", nargs="?", type=Path,
                        help="raw .jsonl file (default: newest in data/raw)")
    args = parser.parse_args()

    raw_path = args.snapshot or latest_snapshot()
    df, counts = clean(load_raw(raw_path))
    summary = {"source_snapshot": raw_path.name, **summarise(df, counts)}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # adzuna_gb_2026-10-24.jsonl -> postings_gb_2026-10-24.csv
    stem = raw_path.stem.replace("adzuna_", "postings_", 1)
    out_path = PROCESSED_DIR / f"{stem}.csv"
    df.to_csv(out_path, index=False)
    with open(PROCESSED_DIR / f"{stem}.meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved {len(df)} postings to {out_path}")


if __name__ == "__main__":
    main()
