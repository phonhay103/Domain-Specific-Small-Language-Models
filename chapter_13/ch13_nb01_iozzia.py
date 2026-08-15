"""Implementing a RAG System with Open Source SLMs and LanceDB.

Companion script for chapter 13 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2025).

Implements an offline RAG (Retrieval Augmented Generation) pipeline using small
language models, all-mpnet-base-v2 embeddings, and the LanceDB vector database.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import os
import re
import sys
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import fitz
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import requests
import torch
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English
from transformers import AutoTokenizer

# Common functional & UI utilities
from common.functional import chunk_list
from common.ui import (
    STYLE_INDEX,
    STYLE_NUMBER,
    STYLE_PRIMARY,
    STYLE_SECONDARY,
    STYLE_SUCCESS,
    STYLE_TEXT,
    STYLE_WARNING,
    console,
    create_table,
    pause,
    render_banner,
    render_card,
    render_step,
    render_takeaways,
    status_spinner,
)


# ---------------------------------------------------------------------------
# Immutable Domain Records & Constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DocumentChunk:
    """Immutable text chunk extracted from source document."""

    page_number: int
    text: str
    token_count: int


@dataclass(frozen=True)
class RetrievalRecord:
    """Immutable search match returned from LanceDB vector query."""

    rank: int
    page_number: int
    distance: float
    text_content: str


PDF_URL = "https://arxiv.org/pdf/2401.08671"
PDF_PATH = "2401.08671.pdf"
EMBEDDINGS_SAVE_PATH = "text_chunks_and_embeddings_df.csv"
LANCEDB_PATH = "paperdb"
LANCEDB_TABLE = "paper_embeddings_table"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
GENERATIVE_MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct-gguf"
GENERATIVE_MODEL_FILE = "*-q4.gguf"
GENERATIVE_TOKENIZER_ID = "microsoft/Phi-3-mini-4k-instruct"
QUERY = "blocked KV-cache"
NUM_SENTENCE_CHUNK_SIZE = 10
MIN_TOKEN_LENGTH = 30
TOP_K_RESULTS = 3
LLM_N_CTX = 1024
EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Pure Functions & Preprocessing
# ---------------------------------------------------------------------------
def normalize_extracted_text(text: str) -> str:
    """Pure string cleaner: collapses newlines and strips whitespace."""
    return text.replace("\n", " ").strip()


def build_rag_prompt(query: str, context_items: Sequence[RetrievalRecord], tokenizer: Any) -> str:
    """Pure RAG prompt constructor with chat template grounding."""
    context = "- " + "\n- ".join([item.text_content for item in context_items])
    base_prompt = (
        "Based on the following context items, please answer the query.\n"
        "Make sure your answers are as explanatory as possible.\n\n"
        f"Context Items:\n{context}\n\n"
        f"User Query: {query}\n"
        "Answer:"
    )
    dialogue = [{"role": "user", "content": base_prompt}]
    return str(tokenizer.apply_chat_template(conversation=dialogue, tokenize=False, add_generation_prompt=True))


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_retrieval_table(results: Sequence[RetrievalRecord]) -> None:
    """Render LanceDB nearest neighbor search matches."""
    columns = [
        ("Rank", STYLE_NUMBER, "center"),
        ("Cosine Distance", STYLE_WARNING, "right"),
        ("Page", STYLE_SECONDARY, "center"),
        ("Retrieved Context Snippet", STYLE_TEXT, "left"),
    ]
    rows = [
        (
            str(r.rank),
            f"{r.distance:.4f}",
            f"p. {r.page_number}",
            r.text_content[:100] + "...",
        )
        for r in results
    ]
    console.print(create_table("LanceDB Vector Retrieval Matches", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute local RAG pipeline: ingest -> embed -> index -> retrieve -> generate."""
    render_banner(
        title="Local RAG with Open-Source SLMs & LanceDB Vector Store",
        subtitle="Chapter 13: Domain-Specific Small Language Models",
        metadata={
            "Embeddings Model": EMBEDDING_MODEL_NAME,
            "Vector Database": "LanceDB (Serverless Columnar)",
            "Query": f'"{QUERY}"',
        },
        icon="🚀",
    )

    # Step 1: Document Ingestion & Chunking
    render_step(1, "PDF Ingestion & Sentence Extraction", icon="📋")
    if not os.path.exists(PDF_PATH):
        with status_spinner(f"Downloading paper from '{PDF_URL}'..."):
            response = requests.get(PDF_URL)
            with open(PDF_PATH, "wb") as f:
                f.write(response.content)

    doc = fitz.open(PDF_PATH)
    nlp = English()
    nlp.add_pipe("sentencizer")

    raw_chunks: list[DocumentChunk] = []
    for page_idx, page in enumerate(doc, start=1):
        clean_text = normalize_extracted_text(page.get_text())
        sentences = [str(s) for s in nlp(clean_text).sents]
        for sentence_group in chunk_list(sentences, NUM_SENTENCE_CHUNK_SIZE):
            joined = "".join(sentence_group).replace("  ", " ").strip()
            joined = re.sub(r"\.([A-Z])", r". \1", joined)
            token_count = int(len(joined) / 4)
            if token_count > MIN_TOKEN_LENGTH:
                raw_chunks.append(DocumentChunk(page_number=page_idx, text=joined, token_count=token_count))

    render_card(
        title="Ingestion Statistics",
        content=(
            f"[text.muted]Total PDF Pages:[/text.muted] [brand.secondary]{len(doc)}[/brand.secondary]\n"
            f"[text.muted]Extracted Dense Chunks (> {MIN_TOKEN_LENGTH} tokens):[/text.muted] [text.highlight]{len(raw_chunks)}[/text.highlight]"
        ),
        icon="📄",
    )

    # Step 2: Dense Embedding & LanceDB Indexing
    render_step(2, "Generating Dense Vector Embeddings & LanceDB Indexing", icon="🧠")
    with status_spinner(f"Loading '{EMBEDDING_MODEL_NAME}'..."):
        embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME, device="cpu")

    with status_spinner("Encoding text chunks into 768-dimensional dense vectors..."):
        texts = [c.text for c in raw_chunks]
        embeddings = embedding_model.encode(texts, show_progress_bar=False)

    records = [
        {"page_number": c.page_number, "sentence_chunk": c.text, "vector": emb.tolist()}
        for c, emb in zip(raw_chunks, embeddings)
    ]
    df_lancedb = pd.DataFrame(records)

    db = lancedb.connect(LANCEDB_PATH)
    custom_schema = pa.schema(
        [
            pa.field("page_number", pa.int64()),
            pa.field("sentence_chunk", pa.utf8()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ]
    )
    tbl = db.create_table(LANCEDB_TABLE, schema=custom_schema, mode="overwrite")
    tbl.add(df_lancedb)
    render_card(
        "Vector Table Created",
        f"LanceDB table [text.highlight]{LANCEDB_TABLE}[/text.highlight] created with [brand.secondary]{len(raw_chunks)} vectors[/brand.secondary].",
        icon="💾",
    )

    # Step 3: LanceDB Vector Retrieval
    render_step(3, "Executing Cosine Similarity Vector Retrieval", icon="🔍")
    query_vector = embedding_model.encode(QUERY, convert_to_tensor=False)
    results = tbl.search(query_vector).limit(TOP_K_RESULTS).to_pandas()

    retrieval_records = tuple(
        RetrievalRecord(
            rank=i,
            page_number=int(row["page_number"]),
            distance=float(row["_distance"]),
            text_content=str(row["sentence_chunk"]),
        )
        for i, (_, row) in enumerate(results.iterrows(), start=1)
    )
    render_retrieval_table(retrieval_records)

    # Step 4: RAG Augmented Generation
    render_step(4, "Grounded RAG Answer Generation with Phi-3 Mini SLM", icon="✨")
    try:
        from llama_cpp import Llama

        with status_spinner(f"Loading GGUF SLM '{GENERATIVE_MODEL_REPO}'..."):
            llm = Llama.from_pretrained(
                repo_id=GENERATIVE_MODEL_REPO, filename=GENERATIVE_MODEL_FILE, verbose=False, n_ctx=LLM_N_CTX
            )
            gen_tokenizer = AutoTokenizer.from_pretrained(GENERATIVE_TOKENIZER_ID)

        prompt = build_rag_prompt(QUERY, retrieval_records[:2], gen_tokenizer)
        with status_spinner("Generating grounded response..."):
            output = llm(prompt, max_tokens=128, stop=["Q:", "\n"], echo=False)
            answer_text = output["choices"][0]["text"].strip()

        render_card("RAG Generated Answer", answer_text, icon="✔")
    except (ImportError, Exception):
        render_card("Inference Status", "GGUF LLM execution requires llama-cpp-python and model weights.", icon="ℹ️")

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Local Serverless RAG",
                "Pairing lightweight embedding models (all-mpnet-base-v2) with LanceDB and quantized GGUF models allows building 100% private, offline RAG pipelines on consumer laptops without API dependencies.",
            ),
            (
                "Grounded Prompt Formatting",
                "Structuring prompts with explicit context constraints prevents SLMs from confabulating facts when answers are missing from retrieved passages.",
            ),
            (
                "IVF-FLAT Indexing",
                "Inverted File Index partitions vectors into clusters, searching only the most promising clusters at query time to keep retrieval times under a few milliseconds.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
