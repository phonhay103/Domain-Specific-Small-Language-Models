"""FAISS for Text – Dense Vector Similarity Search.

Companion script for Chapter 2 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Demonstrates dense vector search and semantic similarity indexing using FAISS
(Facebook AI Similarity Search) and SentenceTransformers embeddings.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Common functional & UI utilities
from common.functional import map_tuple
from common.ui import (
    STYLE_INDEX,
    STYLE_NUMBER,
    STYLE_SECONDARY,
    STYLE_TEXT,
    STYLE_WARNING,
    console,
    create_table,
    pause,
    render_banner,
    render_card,
    render_device_info,
    render_step,
    render_takeaways,
    status_spinner,
)


# ---------------------------------------------------------------------------
# Immutable Domain Records (Functional Data Structures)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Document:
    """Immutable representation of a text document."""

    context: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    """Immutable representation of a similarity search result."""

    rank: int
    corpus_id: int
    distance: float
    document: Document


# ---------------------------------------------------------------------------
# Immutable Constants & Dataset
# ---------------------------------------------------------------------------
MODEL_ID = "paraphrase-mpnet-base-v2"
SEARCH_QUERY = "He throws webs"
TOP_K = 4

CORPUS: tuple[Document, ...] = (
    Document("spiderman", "His secret identity is Peter Parker"),
    Document("ironman", "A businessman and engineer who runs the company Stark Industries"),
    Document("spiderman", "Superhuman spider-powers and abilities after being bitten by a radioactive spider"),
    Document(
        "captainamerica",
        "A frail man enhanced to the peak of human physical perfection by an experimental super-soldier serum",
    ),
)


# ---------------------------------------------------------------------------
# Pure Transformation Functions
# ---------------------------------------------------------------------------
def extract_texts(documents: Sequence[Document]) -> tuple[str, ...]:
    """Pure transformation: extract text sequence from documents."""
    return map_tuple(lambda doc: doc.text, documents)


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Construct an exact L2 FAISS index populated with embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def perform_similarity_search(
    index: faiss.IndexFlatL2,
    query_vector: np.ndarray,
    corpus: Sequence[Document],
    top_k: int = TOP_K,
) -> tuple[SearchResult, ...]:
    """Pure query projection: search vector index and construct immutable SearchResult records."""
    distances, indices = index.search(query_vector, k=top_k)
    return tuple(
        SearchResult(
            rank=rank,
            corpus_id=int(idx),
            distance=float(dist),
            document=corpus[idx],
        )
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0], strict=False), start=1)
    )


# ---------------------------------------------------------------------------
# Functional View / Rendering Helpers
# ---------------------------------------------------------------------------
def render_corpus_table(corpus: Sequence[Document]) -> None:
    """Render the corpus document table using soothing palette."""
    columns = [
        ("Index", STYLE_INDEX, "right"),
        ("Category", STYLE_SECONDARY, "left"),
        ("Document Text", STYLE_TEXT, "left"),
    ]
    rows = [(i, doc.context, doc.text) for i, doc in enumerate(corpus)]
    console.print(create_table("Corpus Documents", columns, rows))
    pause()


def render_search_results_table(query: str, results: Sequence[SearchResult]) -> None:
    """Render the search results ranking table."""
    columns = [
        ("Rank", STYLE_NUMBER, "center"),
        ("Corpus ID", STYLE_INDEX, "right"),
        ("L2 Distance", STYLE_WARNING, "right"),
        ("Category", STYLE_SECONDARY, "left"),
        ("Matching Content", STYLE_TEXT, "left"),
    ]
    rows = [
        (res.rank, res.corpus_id, f"{res.distance:.4f}", res.document.context, res.document.text) for res in results
    ]
    console.print(create_table(f"Semantic Search Results for: '{query}'", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute the functional FAISS text similarity pipeline."""
    render_banner(
        title="FAISS Dense Vector Similarity Search",
        subtitle="Chapter 2: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Query": f'"{SEARCH_QUERY}"',
            "Metric": "Euclidean L2",
        },
        icon="🔍",
    )

    # Step 1: Display Corpus
    render_step(1, "Displaying Immutable Corpus Documents", icon="📋")
    render_corpus_table(CORPUS)

    # Step 2: Encode and Build Index
    render_step(2, "Generating Dense Embeddings & Building Index", icon="🧠")
    with status_spinner(f"Loading SentenceTransformer '{MODEL_ID}'..."):
        model = SentenceTransformer(MODEL_ID)
    render_device_info(model.device, model=model)

    corpus_texts = extract_texts(CORPUS)
    with status_spinner("Encoding corpus texts into 768-dimensional embeddings..."):
        embeddings = model.encode(list(corpus_texts))
        index = create_faiss_index(embeddings)

    render_card(
        title="Vector Index Status",
        content=(
            f"[text.muted]Indexed Vectors:[/text.muted] [text.highlight]{index.ntotal}[/text.highlight]\n"
            f"[text.muted]Embedding Dimension:[/text.muted] [text.highlight]{embeddings.shape[1]}[/text.highlight]\n"
            f"[text.muted]Index Type:[/text.muted] [text.main]IndexFlatL2 (Exact brute-force L2 distance)[/text.main]"
        ),
        icon="✔",
    )

    # Step 3: Perform Vector Similarity Search
    render_step(3, "Executing Semantic Vector Search", icon="⚡")
    with status_spinner(f"Searching nearest neighbors for '{SEARCH_QUERY}'..."):
        query_vector = model.encode([SEARCH_QUERY])
        results = perform_similarity_search(index, query_vector, CORPUS, top_k=TOP_K)

    render_search_results_table(SEARCH_QUERY, results)

    # Top match highlight card
    best_match = results[0]
    render_card(
        title="Top Semantic Match",
        content=(
            f'[text.highlight]"{best_match.document.text}"[/text.highlight]\n\n'
            f"[text.muted]Category:[/text.muted] [brand.secondary]{best_match.document.context}[/brand.secondary]  •  "
            f"[text.muted]L2 Distance:[/text.muted] [status.success]{best_match.distance:.4f}[/status.success]"
        ),
        icon="🎯",
        border_style="#585b70",
    )

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Semantic vs Keyword Search",
                "The query 'He throws webs' matches Spiderman documents even though the exact words 'throws' or 'webs' do not appear in the text.",
            ),
            (
                "Distance Metric",
                "FAISS uses Euclidean L2 distance (lower = more semantically related). The closest vector correctly identifies Spiderman's powers.",
            ),
            (
                "IndexFlatL2 Trade-off",
                "FlatL2 provides exact 100% recall (brute-force search). For production at scale (millions of vectors), approximate indexes like IVF-FLAT or HNSW trade slight accuracy for millisecond speed.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
