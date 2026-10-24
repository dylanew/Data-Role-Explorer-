"""Find skills in job postings with a curated dictionary, and count them.

Usage (from the repo root):
    python -m src.skills                                         # newest file
    python -m src.skills data/processed/postings_gb_2026-10-24.csv

What it does, in order:
    1. Load the cleaned postings written by src/clean.py.
    2. Look for every skill in SKILLS in each posting's title and
       description, using its list of synonyms ("scikit-learn", "sklearn").
    3. Write one row per (posting, skill) found, with the exact text that
       matched, so every match can be checked by eye.
    4. Count how many in-scope postings mention each skill, overall and per
       title family, and how many postings have no skills at all.

Outputs, next to the cleaned postings in data/processed/:
    skills_<country>_<date>.csv           one row per posting + skill
    skill_frequency_<country>_<date>.csv  one row per skill, with counts
    skills_<country>_<date>.meta.json     summary numbers for this run

Design notes:
    - The dictionary is plain text below, grouped by category, so it is easy
      to read and change. Each skill has a display name and a list of the
      ways a job ad might write it.
    - A synonym matches only as a whole word: "scala" does not match
      "scalable", and "excel" does not match "excellent".
    - Spaces and hyphens inside a synonym are interchangeable, so
      "scikit learn", "scikit-learn" and "power-bi" all match.
    - Matching ignores case, except for a few short words that are also
      ordinary words or letters (see CASE_SENSITIVE).
    - The title and the description are searched separately, and both
      results are kept. Frequencies count description matches only, because
      the title is what we later compare clusters against: if "Machine
      Learning Engineer" in the title counted as the skill "machine
      learning", the comparison would be partly circular.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# --- The skill dictionary ----------------------------------------------------
#
# {category: {skill name: [synonyms]}}. Synonyms are written in lowercase;
# the skill name itself is not matched unless it is also in the list.

SKILLS = {
    "programming": {
        "Python": ["python"],
        "R": [],                      # needs its own rule, see SPECIAL_PATTERNS
        "SQL": ["sql", "t-sql", "tsql", "pl/sql"],
        "Scala": ["scala"],
        "Java": ["java"],
        "C++": ["c++"],
        "C#": ["c#"],
        "Go": ["golang"],             # "Go" alone is too often the verb
        "JavaScript / TypeScript": ["javascript", "typescript", "node.js", "nodejs"],
        "SAS": ["SAS"],
        "VBA": ["vba"],
        "MATLAB": ["matlab"],
    },
    "data engineering": {
        "Spark": ["spark", "pyspark", "spark sql"],
        "Databricks": ["databricks"],
        "Airflow": ["airflow"],
        "dbt": ["dbt"],
        "Kafka": ["kafka"],
        "Hadoop": ["hadoop", "hive"],
        "ETL / ELT": ["etl", "elt"],
        "Data pipelines": ["data pipeline", "data pipelines"],
        "Data modelling": ["data modelling", "data modeling", "dimensional modelling",
                           "dimensional modeling", "kimball"],
        "Data warehousing": ["data warehouse", "data warehouses", "data warehousing"],
        "Data lake / lakehouse": ["data lake", "data lakes", "lakehouse", "delta lake"],
        "SSIS": ["ssis"],
        "Azure Data Factory": ["azure data factory", "adf"],
        "Fivetran": ["fivetran"],
        "Informatica": ["informatica"],
    },
    "databases & warehouses": {
        "Snowflake": ["snowflake"],
        "BigQuery": ["bigquery", "big query"],
        "Redshift": ["redshift"],
        "Azure Synapse": ["synapse"],
        "Microsoft Fabric": ["microsoft fabric", "ms fabric", "Fabric"],
        "SQL Server": ["sql server", "mssql"],
        "PostgreSQL": ["postgresql", "postgres"],
        "MySQL": ["mysql"],
        "Oracle": ["oracle"],
        "MongoDB": ["mongodb", "mongo"],
        "NoSQL": ["nosql"],
    },
    "cloud & devops": {
        "AWS": ["aws", "amazon web services"],
        "Azure": ["azure"],
        "GCP": ["gcp", "google cloud"],
        "Docker": ["docker"],
        "Kubernetes": ["kubernetes", "k8s"],
        "Terraform": ["terraform"],
        "CI/CD": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
        "Git": ["git", "github", "gitlab"],
        "Linux": ["linux", "unix"],
        "APIs": ["api", "apis", "rest api", "restful"],
    },
    "BI & visualisation": {
        "Excel": ["Excel", "ms excel", "microsoft excel"],
        "Power BI": ["power bi", "powerbi"],
        "Tableau": ["tableau"],
        "Looker": ["looker"],
        "Qlik": ["qlik", "qlikview", "qlik sense"],
        "DAX": ["dax"],
        "SSRS": ["ssrs"],
        "Alteryx": ["alteryx"],
        "Dashboards": ["dashboard", "dashboards"],
        "Data visualisation": ["data visualisation", "data visualization",
                               "data viz", "visualisation", "visualization"],
    },
    "machine learning": {
        "Machine learning": ["machine learning"],
        "Deep learning": ["deep learning", "neural network", "neural networks"],
        "scikit-learn": ["scikit-learn", "sklearn"],
        "PyTorch": ["pytorch", "torch"],
        "TensorFlow / Keras": ["tensorflow", "keras"],
        "XGBoost / LightGBM": ["xgboost", "lightgbm", "catboost"],
        "pandas / NumPy": ["pandas", "numpy"],
        "NLP": ["nlp", "natural language processing"],
        "Computer vision": ["computer vision"],
        "Forecasting / time series": ["forecasting", "time series", "time-series"],
        "Recommender systems": ["recommender system", "recommender systems",
                                "recommendation systems", "recommendation engine"],
    },
    "generative AI": {
        "LLMs": ["llm", "llms", "large language model", "large language models"],
        "Generative AI": ["generative ai", "genai", "gen ai"],
        "RAG": ["retrieval augmented generation", "retrieval-augmented generation"],
        "AI agents": ["agentic", "ai agents", "multi-agent", "llm agents"],
        "Prompt engineering": ["prompt engineering", "prompting"],
        "LangChain / LangGraph": ["langchain", "langgraph", "llamaindex"],
        "Hugging Face": ["hugging face", "huggingface", "transformers"],
        "Fine-tuning": ["fine-tuning", "fine tuning", "finetuning"],
        "Vector databases": ["vector database", "vector databases", "vector store",
                             "pinecone", "weaviate"],
        "OpenAI API": ["openai"],
    },
    "MLOps": {
        "MLOps": ["mlops"],
        "MLflow": ["mlflow"],
        "SageMaker": ["sagemaker"],
        "Vertex AI": ["vertex ai", "vertex"],
        "Azure ML": ["azure ml", "azure machine learning"],
        "Model deployment": ["model deployment", "deploying models", "productionise",
                             "productionize", "model serving"],
    },
    "statistics & methods": {
        "Statistics": ["statistics", "statistical"],
        "A/B testing": ["a/b testing", "a/b test", "a/b tests", "ab testing",
                        "experimentation"],
        "Econometrics": ["econometrics", "econometric"],
        "Bayesian methods": ["bayesian"],
        "Optimisation": ["mathematical optimisation", "mathematical optimization",
                         "operations research", "linear programming"],
    },
    "business & ways of working": {
        "Stakeholder management": ["stakeholder", "stakeholders"],
        "Communication": ["communication skills", "communicate", "communicating"],
        "Agile": ["agile", "scrum"],
        "Data governance": ["data governance"],
        "Data quality": ["data quality"],
        "Master data": ["master data", "mdm"],
        "Salesforce": ["salesforce"],
        "SAP": ["SAP"],
        "JIRA": ["jira", "confluence"],
    },
}

# Synonyms written here are matched with exact case. They are short words
# that are also ordinary English: "SAS" vs "sas", "Excel" vs "excel" (to do
# well), "Fabric" vs "fabric", "SAP" vs "sap". Everything else ignores case.
CASE_SENSITIVE = {"SAS", "Excel", "Fabric", "SAP"}

# Skills that a list of words cannot describe. "R" is a single letter, so we
# only count a capital R standing on its own, not in "R&D", "A&R", a
# reference like "R-2079" or a code like "R(D)SVS". "RAG" is also a
# red/amber/green project status, so "RAG status" or "RAG rating" is skipped.
SPECIAL_PATTERNS = {
    "R": r"(?<![\w&/.-])R(?![\w&/(-])",
    "RAG": r"(?<![A-Za-z0-9])RAG(?![A-Za-z0-9])(?!\s+(?:status|rating|report))",
}


# --- Matching ----------------------------------------------------------------

def synonym_pattern(synonym):
    """Turn one synonym into a regex that only matches it as a whole word.

    "power bi" -> matches "Power BI", "power-bi" and "powerbi"
    "scala"    -> matches "Scala" but not "scalable"
    "c++"      -> matches "C++"
    """
    words = [re.escape(w) for w in re.split(r"[\s-]+", synonym)]
    body = r"[\s-]?".join(words)
    # \b does not work next to symbols like "+" or "#", so we say directly
    # that no letter or digit may come right before or after the match.
    return rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])"


def compile_skills(skills=SKILLS):
    """Build one compiled regex per skill, plus its category.

    Returns a list of (skill, category, [compiled regexes]).
    """
    compiled = []
    for category, entries in skills.items():
        for skill, synonyms in entries.items():
            patterns = []
            insensitive = [s for s in synonyms if s not in CASE_SENSITIVE]
            sensitive = [s for s in synonyms if s in CASE_SENSITIVE]
            if insensitive:
                patterns.append(re.compile(
                    "|".join(synonym_pattern(s) for s in insensitive), re.IGNORECASE))
            if sensitive:
                patterns.append(re.compile("|".join(synonym_pattern(s) for s in sensitive)))
            if skill in SPECIAL_PATTERNS:
                patterns.append(re.compile(SPECIAL_PATTERNS[skill]))
            compiled.append((skill, category, patterns))
    return compiled


COMPILED_SKILLS = compile_skills()


def find_skills(text, compiled=COMPILED_SKILLS):
    """Return {skill: first text that matched it} for one piece of text."""
    if not isinstance(text, str) or not text:
        return {}
    found = {}
    for skill, _category, patterns in compiled:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                found[skill] = match.group(0)
                break
    return found


def extract(df, compiled=COMPILED_SKILLS):
    """One row per (posting, skill) found in its title or description."""
    category_of = {skill: category for skill, category, _ in compiled}
    rows = []
    for posting in df.itertuples(index=False):
        in_title = find_skills(posting.title, compiled)
        in_description = find_skills(posting.description, compiled)
        for skill in sorted(set(in_title) | set(in_description)):
            rows.append({
                "id": posting.id,
                "skill": skill,
                "category": category_of[skill],
                "in_title": skill in in_title,
                "in_description": skill in in_description,
                "matched_text": in_description.get(skill) or in_title.get(skill),
            })
    columns = ["id", "skill", "category", "in_title", "in_description", "matched_text"]
    return pd.DataFrame(rows, columns=columns)


# --- Frequency analysis --------------------------------------------------------

def frequency(postings, matches):
    """How many in-scope postings mention each skill in their description.

    Returns one row per skill: overall count and share, then the share of
    postings in each title family that mention it. Shares are more useful
    than counts for families, because the families differ in size.
    """
    scope = postings[postings["is_data_role"] & ~postings["is_training_ad"]]
    hits = matches[matches["in_description"] & matches["id"].isin(scope["id"])]
    hits = hits.merge(scope[["id", "title_family"]], on="id")

    n_scope = len(scope)
    overall = (hits.groupby(["skill", "category"]).size()
                   .rename("n_postings").reset_index())
    overall["share"] = (overall["n_postings"] / n_scope).round(4)

    family_sizes = scope["title_family"].value_counts()
    by_family = (hits.groupby(["skill", "title_family"]).size()
                     .unstack(fill_value=0)
                     # families with no matches at all still get a column of 0s
                     .reindex(columns=sorted(family_sizes.index), fill_value=0)
                     .div(family_sizes, axis=1)
                     .round(4)
                     .add_prefix("share_"))
    by_family.columns = [c.replace(" ", "_") for c in by_family.columns]

    return (overall.merge(by_family, left_on="skill", right_index=True)
                   .sort_values(["n_postings", "skill"], ascending=[False, True])
                   .reset_index(drop=True))


def summarise(postings, matches):
    """Coverage numbers, so the sparsity of the snippets is visible."""
    scope = postings[postings["is_data_role"] & ~postings["is_training_ad"]]
    in_desc = matches[matches["in_description"]]
    per_posting = in_desc.groupby("id").size().reindex(scope["id"], fill_value=0)
    any_source = matches.groupby("id").size().reindex(scope["id"], fill_value=0)

    zero_by_family = (scope.assign(n=per_posting.values)
                           .groupby("title_family")["n"]
                           .apply(lambda n: round(float((n == 0).mean()), 3)))
    return {
        "postings_in_scope": len(scope),
        "skills_in_dictionary": len(COMPILED_SKILLS),
        "skills_found_at_least_once": int(in_desc[in_desc["id"].isin(scope["id"])]
                                          ["skill"].nunique()),
        "postings_with_zero_skills": int((per_posting == 0).sum()),
        "share_with_zero_skills": round(float((per_posting == 0).mean()), 3),
        "postings_with_zero_skills_incl_title": int((any_source == 0).sum()),
        "skills_per_posting": {
            "mean": round(float(per_posting.mean()), 2),
            "median": float(per_posting.median()),
            "max": int(per_posting.max()),
        },
        "share_with_zero_skills_by_family": zero_by_family.to_dict(),
    }


def latest_postings(processed_dir=PROCESSED_DIR):
    files = sorted(Path(processed_dir).glob("postings_*.csv"))
    if not files:
        raise SystemExit(f"No cleaned postings in {processed_dir}. Run python -m src.clean first.")
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("postings", nargs="?", type=Path,
                        help="cleaned postings .csv (default: newest in data/processed)")
    args = parser.parse_args()

    path = args.postings or latest_postings()
    postings = pd.read_csv(path, dtype={"id": str})
    matches = extract(postings)
    freq = frequency(postings, matches)
    summary = {"source_postings": path.name, **summarise(postings, matches)}

    # postings_gb_2026-10-24.csv -> skills_gb_2026-10-24.csv
    suffix = path.stem.replace("postings_", "", 1)
    out_dir = path.parent
    matches.to_csv(out_dir / f"skills_{suffix}.csv", index=False)
    freq.to_csv(out_dir / f"skill_frequency_{suffix}.csv", index=False)
    with open(out_dir / f"skills_{suffix}.meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nTop 25 skills (share of in-scope postings, description only):")
    print(freq.head(25)[["skill", "category", "n_postings", "share"]].to_string(index=False))


if __name__ == "__main__":
    main()
