"""Download MovieLens and construct a time-safe, deterministic evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


@dataclass
class Case:
    user_id: int
    target: int
    timestamp: int
    history: pd.DataFrame
    candidates: list[int]


@dataclass
class Dataset:
    movies: pd.DataFrame
    train: pd.DataFrame
    cases: list[Case]
    popular: set[int]
    metadata: dict


def download(root: Path) -> Path:
    folder = root / "ml-latest-small"
    if not all((folder / f).exists() for f in ("ratings.csv", "movies.csv")):
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "ml-latest-small.zip"
        if not archive.exists():
            temporary = archive.with_suffix(".part")
            urlretrieve(URL, temporary)
            temporary.replace(archive)
        # Read known members only; never extract arbitrary archive paths.
        folder.mkdir(exist_ok=True)
        with ZipFile(archive) as z:
            for name in ("ratings.csv", "movies.csv", "README.txt"):
                (folder / name).write_bytes(z.read(f"ml-latest-small/{name}"))
    return folder


def prepare(root: Path, users: int = 20, candidates: int = 50,
            catalog_size: int = 1000, seed: int = 42) -> Dataset:
    folder = download(root)
    ratings = pd.read_csv(folder / "ratings.csv")
    movies = pd.read_csv(folder / "movies.csv").set_index("movieId")
    # Shared temporal cutoff: neither catalog selection nor CF sees future rows.
    cutoff = int(ratings.timestamp.quantile(0.8))
    early = ratings[ratings.timestamp < cutoff]
    counts = early.groupby("movieId").size().sort_index().sort_values(
        ascending=False, kind="stable")
    catalog = counts.head(catalog_size).index
    movies = movies.loc[catalog].sort_index()
    train = early[early.movieId.isin(catalog)]
    active = train.groupby("userId").size()
    train = train[train.userId.isin(active[active >= 20].index)].copy()
    popularity = train.groupby("movieId").size().reindex(movies.index, fill_value=0)
    popular = set(popularity.sort_values(ascending=False, kind="stable").head(
        max(1, int(np.ceil(len(movies) * 0.1)))).index)
    catalog_ids = movies.index.to_numpy()
    eligible = []
    for uid, rows in ratings.groupby("userId", sort=True):
        rows = rows.sort_values(["timestamp", "movieId"])
        target = rows.iloc[-1]  # Latest overall, never substitute an older catalog item.
        if target.timestamp < cutoff or target.movieId not in movies.index:
            continue
        past = rows[rows.timestamp < target.timestamp]
        history = past[past.movieId.isin(movies.index)].copy()
        unseen = np.setdiff1d(catalog_ids, past.movieId.to_numpy())
        distractors = unseen[unseen != int(target.movieId)]
        if len(history) < 20 or len(distractors) < candidates - 1:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(uid)]))
        choices = [int(target.movieId), *map(int, rng.choice(
            distractors, size=candidates - 1, replace=False))]
        rng.shuffle(choices)
        eligible.append(Case(int(uid), int(target.movieId), int(target.timestamp),
                             history, choices))
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(eligible), size=min(users, len(eligible)), replace=False)
    cases = [eligible[i] for i in sorted(selected)]
    if not cases:
        raise ValueError("No eligible test users; check dataset and candidate count.")
    digest = hashlib.sha256((folder / "ratings.csv").read_bytes()).hexdigest()
    movie_digest = hashlib.sha256((folder / "movies.csv").read_bytes()).hexdigest()
    return Dataset(movies, train, cases, popular, {
        "source": URL, "ratings_sha256": digest, "movies_sha256": movie_digest,
        "raw_ratings": len(ratings), "raw_users": int(ratings.userId.nunique()),
        "cutoff_timestamp": cutoff, "catalog_size": len(movies),
        "train_ratings": len(train), "train_users": int(train.userId.nunique()),
        "eligible_users": len(eligible), "evaluated_users": len(cases),
        "candidates": candidates, "seed": seed,
        "popular_catalog_items": len(popular),
        "test_targets_below_4_stars": sum(float(ratings[
            (ratings.userId == c.user_id) & (ratings.movieId == c.target)
        ].iloc[-1].rating) < 4 for c in cases),
    })
