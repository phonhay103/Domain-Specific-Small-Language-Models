"""
FAISS for Text – Quick Start
Companion script for Chapter 2 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Demonstrates building a FAISS L2 index from sentence embeddings and
performing a nearest-neighbour similarity search.
"""

# Extracted from CH02_NB01_Iozzia.ipynb

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "paraphrase-mpnet-base-v2"
SEARCH_QUERY = "He throws webs"

CORPUS: list[list[str]] = [
    ["His secret identity is Peter Parker", "spiderman"],
    [
        "A businessman and engineer who runs the company Stark Industries",
        "ironman",
    ],
    [
        "Superhuman spider-powers and abilities after being bitten by a radioactive spider",
        "spiderman",
    ],
    [
        "A frail man enhanced to the peak of human physical perfection"
        " by an experimental super-soldier serum",
        "captainamerica",
    ],
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def build_index(texts: list[str], model: SentenceTransformer) -> faiss.IndexFlatL2:
    """Encode *texts* and insert normalised vectors into a FAISS L2 index."""
    vectors = model.encode(texts).astype(np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def search(
    query: str,
    index: faiss.IndexFlatL2,
    model: SentenceTransformer,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (distances, indices) for the top-*k* nearest neighbours of *query*."""
    query_vector = model.encode([query]).astype(np.float32)
    faiss.normalize_L2(query_vector)
    return index.search(query_vector, k=k)


def format_results(
    distances: np.ndarray,
    indices: np.ndarray,
    corpus_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge raw FAISS results with the original corpus for display."""
    results_df = pd.DataFrame({"distances": distances[0], "ann": indices[0]})
    return pd.merge(results_df, corpus_df, left_on="ann", right_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    corpus_df = pd.DataFrame(CORPUS, columns=["text", "context"])

    model = SentenceTransformer(MODEL_NAME)
    index = build_index(corpus_df["text"].to_list(), model)

    distances, indices = search(SEARCH_QUERY, index, model, k=index.ntotal)

    merged_df = format_results(distances, indices, corpus_df)
    print(merged_df.head())


if __name__ == "__main__":
    main()
