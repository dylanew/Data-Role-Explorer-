"""Turn job postings into numeric features that clustering can use.

Usage (from the repo root):
    python -m src.embed                                   # newest cleaned file, all feature sets
    python -m src.embed --text masked --features tfidf    # one text version, TF-IDF only
    python -m src.embed --model all-mpnet-base-v2         # a different sentence model

What it does, in order:
    1. Load the cleaned postings written by src/clean.py and keep the ones in
       scope (data roles that are not training-course ads).
    2. Build the text to cluster on, in one of three versions (TEXT_VERSIONS).
    3. Turn that text into vectors in one or both of two ways:
         - TF-IDF: counts of words and word pairs, weighted so words that
           appear in every ad count for little, then squeezed from ~10,000
           columns to 100 with truncated SVD (also called LSA).
         - Sentence embeddings: a pretrained sentence-transformers model reads
           the whole text and returns one vector that captures its meaning,
           so "build data pipelines" and "develop ETL workflows" end up close
           together even though they share no words.
    4. Save each feature set, with the posting ids in the same order.

Outputs, in data/processed/:
    features_<method>_<text>_<country>_<date>.npz   ids + feature matrix
    features_<country>_<date>.meta.json              settings used for every file
    tfidf_<text>_<country>_<date>.joblib             fitted TF-IDF + SVD, so the
                                                     app can place new text
                                                     (a pasted CV) in the same space

Design notes:
    - Every vector is scaled to length 1. Then the ordinary (Euclidean)
      distance that K-means uses ranks pairs the same way as cosine distance,
      which is the usual measure for text: it compares what a text is about,
      not how long it is.
    - Seeds are fixed (SEED) wherever there is randomness, so a rerun gives
      the same numbers.
    - Embeddings are the slow step (about a minute on a laptop CPU for 1,857
      short texts). That is why it runs here, offline, and the app only
      loads the saved file.
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# --- Which text to cluster on --------------------------------------------------
#
# "title+description": the job title followed by the snippet.
# "description":       the snippet only.
# "masked":            the snippet with company / contract boilerplate
#                      sentences removed and role names blanked out.
#
# Why "masked" exists: we later compare clusters with job titles. If the text
# says "data engineer", clusters can simply learn the title back, and the
# comparison tells us nothing. Blanking role names forces the clusters to come
# from what the ad says about the work.
TEXT_VERSIONS = ("title+description", "description", "masked")

# Role names to blank out: an optional qualifier ("data", "ML", ...) followed
# by a role noun ("scientist", "engineer", ...). "Data pipelines" is kept,
# because "pipelines" is not a role noun.
ROLE_NAMES = re.compile(
    r"\b(?:(?:data|ml|ai|machine learning|bi|business intelligence|analytics|"
    r"insights?|research|software|platform|reporting)\s+)?"
    r"(?:scientists?|engineers?|analysts?|engineering|science)\b",
    re.IGNORECASE,
)

# A sentence that contains any of these is treated as boilerplate: it is about
# the employer, the pay or the contract, not about the work. The list came
# from reading the top terms of the first TF-IDF clusters, where whole
# clusters formed around "our mission", "inside IR35" and "hybrid working".
BOILERPLATE = re.compile(
    r"mission|cultur|values|\bwe are\b|\bwe're\b|"
    r"\bour (?:company|client|clients|customers|people|purpose|business|"
    r"organisation|organization|team)\b|\bjoin (?:us|our)|\bworld\b|global|award|"
    r"benefit|salary|£|\$|per (?:day|annum|hour)|ir35|hybrid|remote|"
    r"days? (?:a|per) week|office|\blocation|permanent|contract|"
    r"equal opportunit|divers|inclusi|apply|recruit|on behalf|client of ours|"
    r"requisition|\bjob id\b|founded|headquarter|leading provider|market leader|"
    r"employer|career",
    re.IGNORECASE,
)

# Split after . ! ? or around list bullets that some ads use instead.
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\s+[|•·]\s+")


def drop_boilerplate(text):
    """Keep only the sentences that do not look like employer boilerplate."""
    sentences = SENTENCE_BREAK.split(text)
    return " ".join(s for s in sentences if not BOILERPLATE.search(s))


def mask_roles(text):
    return re.sub(r"\s+", " ", ROLE_NAMES.sub(" ", text)).strip()


def build_text(postings, version):
    """One string per posting, in the chosen TEXT_VERSIONS form."""
    description = postings["description"].fillna("")
    if version == "title+description":
        return postings["title"].fillna("") + ". " + description
    if version == "description":
        return description
    if version == "masked":
        return description.map(drop_boilerplate).map(mask_roles)
    raise ValueError(f"unknown text version {version!r}; choose from {TEXT_VERSIONS}")


def in_scope(postings):
    """The postings we cluster: data roles, not training-course ads."""
    keep = postings["is_data_role"] & ~postings["is_training_ad"]
    return postings[keep].reset_index(drop=True)


# --- Features ------------------------------------------------------------------

def make_vectorizer():
    """TF-IDF settings, kept in one place so cluster.py can reuse them.

    - stop_words: drop "the", "and", ...
    - ngram_range (1, 2): single words and pairs, so "machine learning" and
      "data pipelines" count as terms of their own.
    - min_df 3: ignore words in fewer than 3 ads (typos, company names).
    - max_df 0.5: ignore words in more than half the ads ("data", "role").
    - sublinear_tf: a word said 3 times counts less than 3x a word said once.
    """
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                           min_df=3, max_df=0.5, sublinear_tf=True)


def tfidf_features(texts, n_components=100):
    """TF-IDF, reduced to n_components columns with SVD, rows scaled to length 1.

    Returns (features, fitted vectorizer, fitted svd).
    Why reduce: 10,000 sparse columns make distances noisy (most pairs of
    short ads share no words at all). SVD merges words that tend to appear
    together into 100 "topics", which also makes clustering much faster.
    """
    vectorizer = make_vectorizer()
    counts = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    features = normalize(svd.fit_transform(counts))
    return features.astype(np.float32), vectorizer, svd


def embedding_features(texts, model_name=DEFAULT_MODEL):
    """Sentence embeddings from a pretrained sentence-transformers model.

    The import is inside the function so the rest of the project (and the
    tests) work without downloading torch.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    features = model.encode(list(texts), batch_size=64, show_progress_bar=True,
                            normalize_embeddings=True)
    return np.asarray(features, dtype=np.float32)


def model_slug(model_name):
    """'sentence-transformers/all-MiniLM-L6-v2' -> 'all-minilm-l6-v2'."""
    return model_name.split("/")[-1].lower()


def feature_path(out_dir, method, text_version, suffix):
    """features_tfidf_masked_gb_2026-10-24.npz and the like."""
    text = text_version.replace("+", "-")
    return Path(out_dir) / f"features_{method}_{text}_{suffix}.npz"


def load_features(path):
    """Return (ids, matrix) from a file written by save_features."""
    with np.load(path, allow_pickle=False) as data:
        return data["ids"].astype(str), data["X"]


def save_features(path, ids, features):
    np.savez_compressed(path, ids=np.asarray(ids, dtype=str), X=features)


def latest_postings(processed_dir=PROCESSED_DIR):
    files = sorted(Path(processed_dir).glob("postings_*.csv"))
    if not files:
        raise SystemExit(f"No cleaned postings in {processed_dir}. Run python -m src.clean first.")
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("postings", nargs="?", type=Path,
                        help="cleaned postings .csv (default: newest in data/processed)")
    parser.add_argument("--text", nargs="+", choices=TEXT_VERSIONS, default=list(TEXT_VERSIONS),
                        help="which text versions to build features for (default: all)")
    parser.add_argument("--features", nargs="+", choices=["tfidf", "embeddings"],
                        default=["tfidf", "embeddings"])
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"sentence-transformers model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    path = args.postings or latest_postings()
    postings = in_scope(pd.read_csv(path, dtype={"id": str}))
    suffix = path.stem.replace("postings_", "", 1)
    out_dir = path.parent
    ids = postings["id"].to_numpy()

    meta_path = out_dir / f"features_{suffix}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"files": {}}
    meta.update({"source_postings": path.name, "postings": len(postings), "seed": SEED})

    for version in args.text:
        texts = build_text(postings, version)
        if version == "masked":
            print(f"[masked] median length after removing boilerplate: "
                  f"{int(texts.str.len().median())} characters "
                  f"(was {int(postings['description'].str.len().median())})")

        if "tfidf" in args.features:
            features, vectorizer, svd = tfidf_features(texts)
            out = feature_path(out_dir, "tfidf", version, suffix)
            save_features(out, ids, features)
            # stop_words_ lists every word that was cut (~100,000 of them). It
            # is only kept for inspection, and dropping it shrinks the file ~10x.
            vectorizer.stop_words_ = None
            joblib.dump({"vectorizer": vectorizer, "svd": svd},
                        out_dir / f"tfidf_{version.replace('+', '-')}_{suffix}.joblib")
            meta["files"][out.name] = {
                "method": "tfidf", "text": version, "shape": list(features.shape),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "svd_explained_variance": round(float(svd.explained_variance_ratio_.sum()), 3),
                "created": date.today().isoformat(),
            }
            print(f"wrote {out.name} {features.shape}")

        if "embeddings" in args.features:
            features = embedding_features(texts, args.model)
            out = feature_path(out_dir, f"emb-{model_slug(args.model)}", version, suffix)
            save_features(out, ids, features)
            meta["files"][out.name] = {
                "method": "embeddings", "model": args.model, "text": version,
                "shape": list(features.shape), "created": date.today().isoformat(),
            }
            print(f"wrote {out.name} {features.shape}")

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
