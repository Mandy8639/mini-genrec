"""Popularity plus cosine item-to-item collaborative filtering (no learned LLM)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class Baseline:
    def __init__(self, train: pd.DataFrame, movie_ids):
        self.ids = list(map(int, movie_ids))
        self.index = {mid: i for i, mid in enumerate(self.ids)}
        matrix = train.pivot(index="userId", columns="movieId", values="rating")
        # Positive implicit interactions, rather than treating a dislike as affinity.
        matrix = (matrix.reindex(columns=self.ids).fillna(0).to_numpy() >= 4).astype(float)
        similarity = cosine_similarity(matrix.T)
        np.fill_diagonal(similarity, 0)
        # Shrink weak similarities based on the number of common positive users.
        overlap = matrix.T @ matrix
        self.similarity = similarity * overlap / (overlap + 10)
        pop = train.groupby("movieId").size().reindex(self.ids, fill_value=0).to_numpy()
        self.popularity = np.log1p(pop) / max(float(np.log1p(pop).max()), 1)

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
        scores = 0.8 * normalized + 0.2 * self.popularity[indices]
        return [candidates[i] for i in sorted(
            range(len(candidates)), key=lambda i: (-scores[i], candidates[i]))]
