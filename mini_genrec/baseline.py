"""Popularity plus cosine item-to-item collaborative filtering (no learned LLM)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import hashlib
from sklearn.metrics.pairwise import cosine_similarity


class Baseline:
    def __init__(self, train: pd.DataFrame, movie_ids, cf_weight: float = 0.8,
                 popularity_weight: float = 0.2, positive_rating_threshold: float = 4,
                 co_rating_shrinkage: float = 10):
        self.cf_weight = cf_weight
        self.popularity_weight = popularity_weight
        self.ids = list(map(int, movie_ids))
        self.index = {mid: i for i, mid in enumerate(self.ids)}
        matrix = train.pivot(index="userId", columns="movieId", values="rating")
        # Positive implicit interactions, rather than treating a dislike as affinity.
        matrix = (matrix.reindex(columns=self.ids).fillna(0).to_numpy() >= positive_rating_threshold).astype(float)
        similarity = cosine_similarity(matrix.T)
        np.fill_diagonal(similarity, 0)
        # Shrink weak similarities based on the number of common positive users.
        overlap = matrix.T @ matrix
        self.similarity = similarity * overlap / (overlap + co_rating_shrinkage)
        pop = train.groupby("movieId").size().reindex(self.ids, fill_value=0).to_numpy()
        self.popularity = np.log1p(pop) / max(float(np.log1p(pop).max()), 1)

    def rank_popularity(self, candidates: list[int]) -> list[int]:
        return sorted(candidates, key=lambda mid: (-self.popularity[self.index[mid]], mid))

    def rank(self, history: pd.DataFrame, candidates: list[int]) -> list[int]:
        profile = np.zeros(len(self.ids))
        for row in history.itertuples():
            if int(row.movieId) in self.index:
                profile[self.index[int(row.movieId)]] = (float(row.rating) - 3) / 2
        cf = self.similarity @ profile / max(float(np.abs(profile).sum()), 1)
        indices = np.array([self.index[c] for c in candidates])
        values = cf[indices]
        spread = float(np.ptp(values))
        normalized = (values - values.min()) / spread if spread > 0 else values * 0
        scores = self.cf_weight * normalized + self.popularity_weight * self.popularity[indices]
        return [candidates[i] for i in sorted(
            range(len(candidates)), key=lambda i: (-scores[i], candidates[i]))]


def random_rank(candidates: list[int], user_id: int, seed: int = 42) -> list[int]:
    """A stable per-user control, independent of candidate order and its sampling RNG."""
    digest = hashlib.sha256(f"mini-genrec/random/v1:{seed}:{user_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return list(map(int, rng.permutation(sorted(candidates))))
