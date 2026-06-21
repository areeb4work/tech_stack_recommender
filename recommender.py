"""
recommender.py
----------------
Core AI Recommendation Logic for Project 3: Tech Stack Recommender.

Implements the 4-step pipeline taught in the training kit:
  1. Ingestion  -> capture user skills
  2. Scoring    -> TF-IDF vectorization + Cosine Similarity
  3. Sorting    -> rank roles by similarity score
  4. Filtering  -> truncate to Top-N results

This is pure Content-Based Filtering: no historical user data needed,
just item (job role) attributes mapped into the same vector space as
the user's input.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TechStackRecommender:
    def __init__(self, csv_path="raw_skills.csv"):
        # Load the item dataset (job roles + their skill profiles)
        self.df = pd.read_csv(csv_path)

        # Fit the vectorizer ONCE on the job role skill sets.
        # This builds the shared vocabulary space that user input
        # will later be projected into.
        self.vectorizer = TfidfVectorizer()
        self.role_vectors = self.vectorizer.fit_transform(self.df["Skills"])

    def recommend(self, user_skills, top_n=3):
        """
        user_skills: list of strings, e.g. ["Python", "Cloud Computing", "Automation"]
        top_n: how many top matches to return

        Returns a list of dicts: [{"role": ..., "score": ..., "skills": ...}, ...]
        """
        if not user_skills or len(user_skills) == 0:
            return []

        # Step 1: Ingestion - join user skills into a single "document"
        user_text = " ".join(user_skills)

        # Step 2: Scoring - transform user input into the SAME vector
        # space as the job roles (transform, not fit_transform, to avoid
        # creating a mismatched vocabulary)
        user_vector = self.vectorizer.transform([user_text])
        scores = cosine_similarity(user_vector, self.role_vectors).flatten()

        # Attach scores to a working copy of the dataframe
        results = self.df.copy()
        results["score"] = scores

        # Cold-start guard: if every score is zero, there's no real
        # vocabulary overlap between user input and any job role.
        cold_start = bool((results["score"] == 0).all())

        # Step 3: Sorting - descending by similarity score
        results = results.sort_values("score", ascending=False)

        # Step 4: Filtering - truncate to Top-N
        top_results = results.head(top_n)

        output = []
        for _, row in top_results.iterrows():
            output.append({
                "role": row["Role"],
                "score": round(float(row["score"]) * 100, 1),  # as a percentage
                "skills": row["Skills"]
            })

        return {"results": output, "cold_start": cold_start}

    def all_known_skills(self):
        """Return the full set of unique skills across all roles, useful
        for autocomplete / suggestion chips in the UI."""
        skills = set()
        for skill_str in self.df["Skills"]:
            for s in skill_str.split():
                skills.add(s)
        return sorted(skills)


if __name__ == "__main__":
    # Quick manual test from the command line
    engine = TechStackRecommender("raw_skills.csv")
    test_input = ["Python", "Cloud Computing", "Automation"]
    print(f"User skills: {test_input}\n")
    output = engine.recommend(test_input, top_n=3)
    for item in output["results"]:
        print(f"{item['role']:<25} {item['score']}% match")
