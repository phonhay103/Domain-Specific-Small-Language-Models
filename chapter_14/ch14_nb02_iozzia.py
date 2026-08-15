"""
Implementing an Agentic RAG System with Open Source SLMs and SmolAgents.

Companion script for chapter 14 of "Domain-Specific Small Language Models"
by Guglielmo Iozzia, Manning Publications, 2025.

Demonstrates an Agentic RAG (Retrieval Augmented Generation) system using
SmolAgents, LangChain, and LanceDB with Small Language Models. GPU recommended.

# Install the missing requirements before running:
# pip install smolagents langchain lancedb langchain-community rank_bm25 pypdf
#             langchain-huggingface ddgs

# Download the sample PDF:
# curl https://arxiv.org/pdf/2502.12923 --output arxiv_250212923.pdf
"""

import os
import sys
from pathlib import Path
from typing import List

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb
import numpy as np
import pyarrow as pa
import torch
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import Field
from smolagents import CodeAgent, DuckDuckGoSearchTool, Tool, TransformersModel

from common.ui import (
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
# Constants
# ---------------------------------------------------------------------------

PDF_PATH = "/content/arxiv_250212923.pdf"
LANCEDB_PATH = "./lancedb"
TABLE_NAME = "document_embeddings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MAX_NEW_TOKENS = 700
AGENT_MAX_STEPS = 3

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
BM25_K = 3
HYBRID_WEIGHTS = [0.4, 0.6]  # [BM25 weight, semantic weight]

AGENT_QUERY = "Do an hybrid search about Smart Home technologies and then search the web about the same"

# ---------------------------------------------------------------------------
# Document loading & embedding
# ---------------------------------------------------------------------------


def load_and_chunk_pdf(pdf_path: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Load a PDF and split it into overlapping text chunks."""
    if not os.path.exists(pdf_path):
        console.print(f"[yellow]PDF not found at {pdf_path}. Creating sample document...[/yellow]")
        return [
            Document(
                page_content="Smart Home technologies leverage IoT devices, energy management algorithms, and edge intelligence to optimize home comfort and security."
            ),
            Document(
                page_content="Wireless mesh networks like Zigbee and Matter enable seamless interoperability across diverse smart home manufacturers."
            ),
        ]
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def create_lancedb_table(
    db_path: str, table_name: str, docs: list[Document], embeddings: HuggingFaceEmbeddings
) -> lancedb.table.LanceTable:
    """Create a LanceDB table and populate it with document embeddings."""
    with console.status(f"[bold green]Populating LanceDB table '{table_name}' with document vectors..."):
        db = lancedb.connect(db_path)

        schema = pa.schema(
            [
                pa.field("embedding", pa.list_(pa.float32(), list_size=EMBEDDING_DIM)),
                pa.field("text", pa.string()),
            ]
        )

        table = db.create_table(
            table_name,
            schema=schema,
            mode="overwrite",
            data=[
                {
                    "embedding": np.array(embeddings.embed_query(doc.page_content), dtype=np.float32)
                    .flatten()
                    .tolist(),
                    "text": doc.page_content,
                }
                for doc in docs
            ],
        )
    console.print(
        f"[bold green]✔[/bold green] LanceDB table initialized with [bold]{len(docs)}[/bold] embedded chunks."
    )
    return table


# ---------------------------------------------------------------------------
# SmolAgents custom tools
# ---------------------------------------------------------------------------


class BM25SearchTool(Tool):
    """SmolAgents tool that performs BM25 keyword search on local documents."""

    name = "do_bm25_search_on_local_documents"
    description = (
        "Uses text search to retrieve the parts of the documentation that could be most relevant to answer a query."
    )
    inputs = {"query": {"type": "string", "description": "The search query string."}}
    output_type = "string"

    def __init__(self, docs: list[Document], **kwargs):
        super().__init__(**kwargs)
        self.docs = docs

    def forward(self, query: str) -> str:
        retriever = BM25Retriever.from_documents(self.docs, k=BM25_K)
        results = retriever.invoke(query)
        return "\nRetrieved documents:\n" + "".join(
            f"\n\n===== Document {i} =====\n" + doc.page_content for i, doc in enumerate(results)
        )


class SemanticSearchTool(Tool):
    """SmolAgents tool that performs vector (semantic) search on a LanceDB table."""

    name = "semantic_search"
    description = "Performs semantic search on a database of document embeddings."
    inputs = {
        "query": {"type": "string", "description": "The search query string."},
        "top_k": {
            "type": "integer",
            "description": "The number of top results to return.",
            "default": 1,
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, table, embeddings, **kwargs):
        super().__init__(**kwargs)
        self.table = table
        self.embeddings = embeddings

    def forward(self, query: str, top_k: int = 1) -> str:
        """Perform semantic search on the document_embeddings table."""
        query_embedding = np.array(self.embeddings.embed_query(query))
        results = self.table.search(query_embedding, vector_column_name="embedding").limit(top_k).to_df()
        return results.to_string()


class LanceDBVectorSearch(BaseRetriever):
    """LangChain-compatible vector search retriever backed by LanceDB."""

    table: lancedb.table.LanceTable = Field(...)
    embeddings: HuggingFaceEmbeddings = Field(...)

    def __init__(self, table: lancedb.table.LanceTable, embeddings: HuggingFaceEmbeddings):
        super().__init__(table=table, embeddings=embeddings)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Retrieve documents relevant to the query via vector search."""
        query_embedding = np.array(self.embeddings.embed_query(query))

        results = self.table.search(query_embedding, vector_column_name="embedding").limit(2).to_pandas()

        return [Document(page_content=row["text"], metadata={}) for _, row in results.iterrows()]


class HybridSearchTool(Tool):
    """SmolAgents tool combining BM25 keyword search and vector search via LangChain EnsembleRetriever."""

    name = "hybrid_search"
    description = "Performs a hybrid search (BM25 keyword + vector search) on a LanceDB table."
    inputs = {
        "query": {"type": "string", "description": "The search query string."},
        "top_k": {
            "type": "integer",
            "description": "The number of top results to return.",
            "default": 1,
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, bm25_retriever, semantic_retriever, **kwargs):
        super().__init__(**kwargs)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=HYBRID_WEIGHTS,
        )

    def forward(self, query: str, top_k: int = 1) -> str:
        docs = self.ensemble_retriever.get_relevant_documents(query)
        return "\nRetrieved documents:\n" + "".join(
            f"\n\n===== Document {i} =====\n" + doc.page_content for i, doc in enumerate(docs)
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_agent(
    hybrid_search_tool: HybridSearchTool,
    model_id: str,
) -> CodeAgent:
    """Instantiate the SmolLM2 model and configure the CodeAgent."""
    with console.status(f"[bold green]Loading {model_id} for SmolAgents CodeAgent..."):
        model = TransformersModel(
            model_id,
            device_map="auto",
            max_new_tokens=MAX_NEW_TOKENS,
            torch_dtype=torch.float16,
        )
    duckduckgo_search_tool = DuckDuckGoSearchTool()
    custom_tools = [hybrid_search_tool, duckduckgo_search_tool]
    return CodeAgent(
        tools=custom_tools,
        model=model,
        max_steps=AGENT_MAX_STEPS,
        verbosity_level=2,
        add_base_tools=False,
    )


def main() -> None:
    """Load documents, build retrieval tools, create agent, and run a query."""
    render_banner(
        title="Agentic RAG with SmolAgents, LangChain & LanceDB",
        subtitle="Chapter 14: Domain-Specific Small Language Models",
        metadata={
            "SLM Engine": MODEL_ID,
            "Embeddings": EMBEDDING_MODEL,
            "Query": f'"{AGENT_QUERY}"',
        },
        icon="🚀",
    )

    # Step 1: Document Ingestion & Chunking
    render_step(1, "Document Ingestion & Overlapping Chunking", icon="📋")
    docs = load_and_chunk_pdf(PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    render_card(
        "Document Chunks Prepared",
        f"Prepared [text.highlight]{len(docs)}[/text.highlight] overlapping text chunks from source documents.",
        icon="📄",
    )

    # Step 2: Vector Embedding & LanceDB Setup
    render_step(2, "Vector Embedding & Columnar LanceDB Setup", icon="🧠")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    table = create_lancedb_table(LANCEDB_PATH, TABLE_NAME, docs, embeddings)

    # Step 3: Building Hybrid Retriever Tools
    render_step(3, "Configuring Hybrid Retriever Tools (BM25 + Dense Semantic)", icon="🔍")
    bm25_retriever = BM25Retriever.from_documents(docs, k=BM25_K)
    semantic_retriever = LanceDBVectorSearch(table=table, embeddings=embeddings)
    hybrid_search_tool = HybridSearchTool(bm25_retriever, semantic_retriever)

    test_hybrid_query = "Smart Home technologies"
    with status_spinner(f"Testing hybrid retrieval for '{test_hybrid_query}'..."):
        test_results = hybrid_search_tool.forward(test_hybrid_query, top_k=2)

    render_card("Hybrid Retrieval Preview", test_results[:400] + "...", icon="✨")

    # Step 4: Executing Agentic RAG Workflow
    render_step(4, "Executing SmolAgents Autonomous ReAct Routing", icon="🤖")
    agent = build_agent(hybrid_search_tool, MODEL_ID)
    agent_output = agent.run(AGENT_QUERY)

    render_card("Final Agentic RAG Output", str(agent_output), icon="🎯")

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Hybrid Search Synergy",
                "BM25 handles rare technical terminology and exact acronyms, while dense embeddings capture broad conceptual and semantic intent.",
            ),
            (
                "Reciprocal Rank Fusion (RRF)",
                "Normalizes ranking scores across disparate retrieval modalities (keyword + vector) to guarantee top relevant context passages.",
            ),
            (
                "Agentic Decision Routing",
                "An autonomous SLM agent dynamically decides whether to query local LanceDB knowledge or search the live web via DuckDuckGo.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
