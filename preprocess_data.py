from __future__ import annotations

from pathlib import Path
import argparse
import random

import pandas as pd


SUPPORTED_DATASETS = {"ml-100k", "ml-1m"}


def read_movielens_100k(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads original MovieLens 100K.

    Expected files:
        u.data
        u.item

    u.data:
        userId<TAB>movieId<TAB>rating<TAB>timestamp

    u.item:
        movieId|title|release_date|video_release_date|IMDb_URL|
        unknown|Action|Adventure|...|Western
    """
    ratings_path = raw_dir / "u.data"
    movies_path = raw_dir / "u.item"

    if not ratings_path.exists():
        raise FileNotFoundError(f"File not found: {ratings_path}")

    if not movies_path.exists():
        raise FileNotFoundError(f"File not found: {movies_path}")

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )

    genre_columns = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    item_columns = [
        "movieId",
        "title",
        "release_date",
        "video_release_date",
        "IMDb_URL",
    ] + genre_columns

    movies_raw = pd.read_csv(
        movies_path,
        sep="|",
        names=item_columns,
        encoding="latin-1",
    )

    def build_genres(row: pd.Series) -> str:
        genres = [genre for genre in genre_columns if row[genre] == 1]
        return "|".join(genres) if genres else "(no genres listed)"

    movies = movies_raw[["movieId", "title"]].copy()
    movies["genres"] = movies_raw.apply(build_genres, axis=1)

    return ratings, movies


def read_movielens_1m(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads original MovieLens 1M.

    Expected files:
        ratings.dat
        movies.dat

    ratings.dat:
        UserID::MovieID::Rating::Timestamp

    movies.dat:
        MovieID::Title::Genres
    """
    ratings_path = raw_dir / "ratings.dat"
    movies_path = raw_dir / "movies.dat"

    if not ratings_path.exists():
        raise FileNotFoundError(f"File not found: {ratings_path}")

    if not movies_path.exists():
        raise FileNotFoundError(f"File not found: {movies_path}")

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1",
    )

    return ratings, movies


def read_movielens(
    dataset: str,
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selects the appropriate reader based on the dataset.
    """
    if dataset == "ml-100k":
        return read_movielens_100k(raw_dir)

    if dataset == "ml-1m":
        return read_movielens_1m(raw_dir)

    raise ValueError(
        f"Unsupported dataset: {dataset}. "
        f"Valid options: {sorted(SUPPORTED_DATASETS)}"
    )


def filter_positive_interactions(
    ratings: pd.DataFrame,
    min_rating: float | None = None,
) -> pd.DataFrame:
    """
    Converts explicit ratings into implicit interactions.

    If min_rating is None:
        keeps every observed rating as positive.

    If min_rating = 4:
        keeps only ratings >= 4.
    """
    if min_rating is None:
        positives = ratings.copy()
    else:
        positives = ratings[ratings["rating"] >= min_rating].copy()

    return positives


def drop_users_with_few_interactions(
    ratings: pd.DataFrame,
    min_interactions: int = 2,
) -> pd.DataFrame:
    """
    For leave-one-out we need at least two interactions per user:
    one for train and one for test.
    """
    counts = ratings.groupby("userId").size()
    valid_users = counts[counts >= min_interactions].index
    return ratings[ratings["userId"].isin(valid_users)].copy()


def create_mappings(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates compact internal IDs starting from 0.

    Traceability is preserved:

        original userId  -> internal user_idx
        original movieId -> internal item_idx

    The model must use user_idx and item_idx.
    To recover titles, use movie_mapping.csv.
    """
    user_ids = sorted(ratings["userId"].unique())
    movie_ids = sorted(ratings["movieId"].unique())

    user_mapping = pd.DataFrame(
        {
            "userId": user_ids,
            "user_idx": range(len(user_ids)),
        }
    )

    movie_mapping = pd.DataFrame(
        {
            "movieId": movie_ids,
            "item_idx": range(len(movie_ids)),
        }
    )

    movie_mapping = movie_mapping.merge(
        movies,
        on="movieId",
        how="left",
    )

    ratings_mapped = ratings.merge(
        user_mapping,
        on="userId",
        how="left",
    )

    ratings_mapped = ratings_mapped.merge(
        movie_mapping[["movieId", "item_idx"]],
        on="movieId",
        how="left",
    )

    ratings_mapped = ratings_mapped[
        [
            "userId",
            "user_idx",
            "movieId",
            "item_idx",
            "rating",
            "timestamp",
        ]
    ].copy()

    return ratings_mapped, user_mapping, movie_mapping


def leave_one_out_split(
    ratings_mapped: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits in NCF style:

    - For each user, the most recent interaction goes to test.
    - The rest goes to train.

    Timestamp is used as the temporal criterion.
    """
    ratings_sorted = ratings_mapped.sort_values(
        ["user_idx", "timestamp", "item_idx"],
        ascending=[True, True, True],
    )

    test_idx = ratings_sorted.groupby("user_idx").tail(1).index

    test = ratings_sorted.loc[test_idx].copy()
    train = ratings_sorted.drop(test_idx).copy()

    train = train.sort_values(["user_idx", "timestamp", "item_idx"])
    test = test.sort_values(["user_idx"])

    return train, test


def generate_test_negatives(
    train: pd.DataFrame,
    test: pd.DataFrame,
    num_items: int,
    num_negatives: int = 99,
    seed: int = 42,
) -> list[list[int]]:
    """
    Generates negatives for evaluation.

    For each user:
    - identifies their observed positive items;
    - excludes those items;
    - samples num_negatives unobserved items.

    Note:
    In implicit recommendation, a negative means "not observed",
    not necessarily "disliked".
    """
    rng = random.Random(seed)
    all_items = set(range(num_items))

    user_pos_items = (
        pd.concat([train, test])
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )

    negatives = []

    for row in test.itertuples(index=False):
        user = row.user_idx
        positive_item = row.item_idx

        interacted = user_pos_items[user]
        candidates = list(all_items - interacted)

        if len(candidates) < num_negatives:
            raise ValueError(
                f"Internal user {user} has only {len(candidates)} "
                f"negative candidates, but {num_negatives} were requested. "
                f"Reduce --num-negatives."
            )

        sampled = rng.sample(candidates, num_negatives)

        if positive_item in sampled:
            raise RuntimeError(
                "Unexpected error: positive item appeared among negatives."
            )

        negatives.append(sampled)

    return negatives


def save_ncf_files(
    output_dir: Path,
    prefix: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    negatives: list[list[int]],
    user_mapping: pd.DataFrame,
    movie_mapping: pd.DataFrame,
    ratings_mapped: pd.DataFrame,
) -> None:
    """
    Saves NCF-style files with dataset prefix.

    Examples:
        ml-100k.train.rating
        ml-100k.test.rating
        ml-100k.test.negative

        ml-1m.train.rating
        ml-1m.test.rating
        ml-1m.test.negative
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = train[["user_idx", "item_idx", "rating", "timestamp"]]
    test_out = test[["user_idx", "item_idx", "rating", "timestamp"]]

    train_out.to_csv(
        output_dir / f"{prefix}.train.rating",
        sep="\t",
        header=False,
        index=False,
    )

    test_out.to_csv(
        output_dir / f"{prefix}.test.rating",
        sep="\t",
        header=False,
        index=False,
    )

    with open(output_dir / f"{prefix}.test.negative", "w", encoding="utf-8") as f:
        for row, neg_items in zip(test.itertuples(index=False), negatives):
            first_col = f"({row.user_idx},{row.item_idx})"
            line = "\t".join([first_col] + [str(x) for x in neg_items])
            f.write(line + "\n")

    user_mapping.to_csv(output_dir / f"{prefix}.user_mapping.csv", index=False)
    movie_mapping.to_csv(output_dir / f"{prefix}.movie_mapping.csv", index=False)
    ratings_mapped.to_csv(output_dir / f"{prefix}.ratings_mapped.csv", index=False)


def validate_outputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    user_mapping: pd.DataFrame,
    movie_mapping: pd.DataFrame,
    negatives: list[list[int]],
    num_negatives: int,
) -> None:
    """
    Basic validations to detect construction errors.
    """
    assert train["user_idx"].min() == 0
    assert train["item_idx"].min() == 0
    assert test["user_idx"].min() == 0
    assert test["item_idx"].min() >= 0

    assert user_mapping["user_idx"].is_unique
    assert movie_mapping["item_idx"].is_unique
    assert user_mapping["userId"].is_unique
    assert movie_mapping["movieId"].is_unique

    assert len(test) == len(user_mapping)
    assert len(negatives) == len(test)

    for neg_list in negatives:
        assert len(neg_list) == num_negatives
        assert len(set(neg_list)) == num_negatives


def default_raw_dir(dataset: str) -> Path:
    if dataset == "ml-100k":
        return Path("data/raw/ml-100k")

    if dataset == "ml-1m":
        return Path("data/raw/ml-1m")

    raise ValueError(f"Unsupported dataset: {dataset}")


def default_output_dir(dataset: str) -> Path:
    if dataset == "ml-100k":
        return Path("data/processed/ml-100k-custom")

    if dataset == "ml-1m":
        return Path("data/processed/ml-1m-custom")

    raise ValueError(f"Unsupported dataset: {dataset}")


def build_dataset(
    dataset: str,
    raw_dir: Path | None,
    output_dir: Path | None,
    min_rating: float | None,
    num_negatives: int,
    seed: int,
) -> None:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Valid options: {sorted(SUPPORTED_DATASETS)}"
        )

    if raw_dir is None:
        raw_dir = default_raw_dir(dataset)

    if output_dir is None:
        output_dir = default_output_dir(dataset)

    ratings, movies = read_movielens(dataset, raw_dir)

    positives = filter_positive_interactions(
        ratings,
        min_rating=min_rating,
    )

    positives = drop_users_with_few_interactions(
        positives,
        min_interactions=2,
    )

    ratings_mapped, user_mapping, movie_mapping = create_mappings(
        positives,
        movies,
    )

    train, test = leave_one_out_split(ratings_mapped)

    num_items = movie_mapping["item_idx"].nunique()

    negatives = generate_test_negatives(
        train=train,
        test=test,
        num_items=num_items,
        num_negatives=num_negatives,
        seed=seed,
    )

    validate_outputs(
        train=train,
        test=test,
        user_mapping=user_mapping,
        movie_mapping=movie_mapping,
        negatives=negatives,
        num_negatives=num_negatives,
    )

    save_ncf_files(
        output_dir=output_dir,
        prefix=dataset,
        train=train,
        test=test,
        negatives=negatives,
        user_mapping=user_mapping,
        movie_mapping=movie_mapping,
        ratings_mapped=ratings_mapped,
    )

    print("Dataset built successfully.")
    print(f"Dataset: {dataset}")
    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Users: {user_mapping.shape[0]}")
    print(f"Items: {movie_mapping.shape[0]}")
    print(f"Positive interactions: {ratings_mapped.shape[0]}")
    print(f"Train interactions: {train.shape[0]}")
    print(f"Test interactions: {test.shape[0]}")
    print(f"Negatives per user in test: {num_negatives}")

    if min_rating is None:
        print("Positive criterion: all observed ratings")
    else:
        print(f"Positive criterion: rating >= {min_rating}")


def parse_min_rating(value: str) -> float | None:
    """
    Allows:
        --min-rating none
        --min-rating all
        --min-rating 4
    """
    if value.lower() in {"none", "null", "all"}:
        return None

    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Builds an NCF-style dataset from MovieLens 100K or MovieLens 1M, "
            "with reproducible mappings back to original IDs."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-100k",
        choices=sorted(SUPPORTED_DATASETS),
        help="Dataset to process. Default: ml-100k.",
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Directory with the original files. "
            "Default for ml-100k: data/raw/ml-100k. "
            "Default for ml-1m: data/raw/ml-1m."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. "
            "Default for ml-100k: data/processed/ml-100k-custom. "
            "Default for ml-1m: data/processed/ml-1m-custom."
        ),
    )

    parser.add_argument(
        "--min-rating",
        type=parse_min_rating,
        default=None,
        help=(
            "Threshold to consider an interaction positive. "
            "Use 'none' or 'all' to keep every observed rating. "
            "Example: --min-rating 4. Default: none."
        ),
    )

    parser.add_argument(
        "--num-negatives",
        type=int,
        default=99,
        help="Number of negatives per user for test. Default: 99.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible negative sampling. Default: 42.",
    )

    args = parser.parse_args()

    build_dataset(
        dataset=args.dataset,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        min_rating=args.min_rating,
        num_negatives=args.num_negatives,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
