"""Match a visitor's skills to a role type (the "match me" page, milestone 6).

How it works:
    1. Find skills in the pasted text with the same dictionary and matching
       rules used on the job ads (src/skills.py). So "sklearn" in a CV counts
       as scikit-learn exactly as it does in an ad.
    2. Build a skill profile for each of the three role types: for every
       skill, the share of that role's ads that mention it. Only skills that
       at least MIN_SHARE of the role's ads mention are kept, so one stray
       mention does not become part of a profile.
    3. Score each role by coverage: the profile shares of the skills you have,
       divided by the profile shares of all its skills. 100% means you have
       every skill in the profile; a skill that 30% of ads ask for counts ten
       times as much as one that 3% ask for.
    4. Missing skills: the profile skills you do not have, most asked-for
       first.

Design notes:
    - Only the three role types are matched (src.cluster.ROLE_CLUSTERS). The
      other five clusters group ads by employer or ad template; "you match
      'Banking employers'" would not tell anyone anything about their skills.
    - Profiles use only ads in the role's own cluster, not ads that got the
      role type from their skills (role_type_source == "skills"). Those were
      labelled with these same skills, so counting them would be circular.
    - Shares are out of all the role's ads, including the many that mention
      no skill at all (the 500-character snippets). That makes the numbers
      small, but they match the cluster profiles page, and coverage divides
      the scale away anyway.
    - No model is loaded: this is a few dictionary look-ups and sums, so the
      app stays light and every score can be explained by hand.
"""

import pandas as pd

from src.skills import find_skills

# Same threshold as src.cluster.signature_skills: a skill must appear in at
# least 2% of a role's ads to be part of its profile.
MIN_SHARE = 0.02


def role_profiles(skills, clusters, role_clusters, min_share=MIN_SHARE):
    """One row per (role cluster, skill) in a profile.

    skills:   the skills_<...>.csv table (one row per posting and skill).
    clusters: the clusters_<...>.csv table (one row per posting).
    Returns columns cluster, cluster_name, skill, category, share, lift, where
    lift is how many times more common the skill is in this role than across
    all postings.
    """
    in_scope = clusters[["id", "cluster", "cluster_name"]]
    hits = skills[skills["in_description"]].merge(in_scope, on="id")
    overall = hits.groupby("skill")["id"].nunique() / len(in_scope)
    category = hits.drop_duplicates("skill").set_index("skill")["category"]

    rows = []
    for cluster in role_clusters:
        members = in_scope[in_scope["cluster"] == cluster]
        share = hits[hits["cluster"] == cluster].groupby("skill")["id"].nunique() / len(members)
        share = share[share >= min_share].sort_values(ascending=False)
        for skill, value in share.items():
            rows.append({"cluster": cluster, "cluster_name": members["cluster_name"].iloc[0],
                         "skill": skill, "category": category[skill],
                         "share": round(value, 4), "lift": round(value / overall[skill], 2)})
    return pd.DataFrame(rows, columns=["cluster", "cluster_name", "skill", "category",
                                       "share", "lift"])


def skills_in_text(text):
    """The dictionary skills found in a piece of text, as a sorted list."""
    return sorted(find_skills(text))


def score_roles(my_skills, profiles):
    """Coverage of each role's profile by my_skills, best match first.

    Returns one row per role: cluster, cluster_name, coverage (0 to 1),
    n_have and n_profile (how many profile skills you have, out of how many).
    """
    mine = set(my_skills)
    rows = []
    for (cluster, name), profile in profiles.groupby(["cluster", "cluster_name"], sort=False):
        have = profile["skill"].isin(mine)
        rows.append({"cluster": cluster, "cluster_name": name,
                     "coverage": profile.loc[have, "share"].sum() / profile["share"].sum(),
                     "n_have": int(have.sum()), "n_profile": len(profile)})
    return pd.DataFrame(rows).sort_values("coverage", ascending=False, kind="stable") \
                             .reset_index(drop=True)


def missing_skills(my_skills, profiles, cluster):
    """The role's profile skills that my_skills does not include, most asked-for first."""
    profile = profiles[profiles["cluster"] == cluster]
    return profile[~profile["skill"].isin(set(my_skills))].reset_index(drop=True)
