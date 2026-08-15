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
from typing import List

import lancedb
import numpy as np
import pyarrow as pa
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import Field
from smolagents import CodeAgent, DuckDuckGoSearchTool, Tool, TransformersModel

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

def load_and_chunk_pdf(pdf_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Load a PDF and split it into overlapping text chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def create_lancedb_table(
    db_path: str, table_name: str, docs: List[Document], embeddings: HuggingFaceEmbeddings
) -> lancedb.table.LanceTable:
    """Create a LanceDB table and populate it with document embeddings."""
    db = lancedb.connect(db_path)

    schema = pa.schema([
        pa.field("embedding", pa.list_(pa.float32(), list_size=EMBEDDING_DIM)),
        pa.field("text", pa.string()),
    ])

    table = db.create_table(
        table_name,
        schema=schema,
        data=[
            {
                "embedding": np.array(
                    embeddings.embed_query(doc.page_content), dtype=np.float32
                ).flatten().tolist(),
                "text": doc.page_content,
            }
            for doc in docs
        ],
    )
    return table

# ---------------------------------------------------------------------------
# SmolAgents custom tools
# ---------------------------------------------------------------------------

class BM25SearchTool(Tool):
    """SmolAgents tool that performs BM25 keyword search on local documents.

    BM25 ranks documents based on term frequency, document length, and term
    rarity — effective for keyword-based information retrieval.
    """

    name = "do_bm25_search_on_local_documents"
    description = (
        "Uses text search to retrieve the parts of the documentation "
        "that could be most relevant to answer a query."
    )
    inputs = {"query": {"type": "string", "description": "The search query string."}}
    output_type = "string"

    def __init__(self, docs: List[Document], **kwargs):
        super().__init__(**kwargs)
        self.docs = docs

    def forward(self, query: str) -> str:
        retriever = BM25Retriever.from_documents(self.docs, k=BM25_K)
        results = retriever.invoke(query)
        return "\nRetrieved documents:\n" + "".join(
            f"\n\n===== Document {i} =====\n" + doc.page_content
            for i, doc in enumerate(results)
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
        results = (
            self.table.search(query_embedding, vector_column_name="embedding")
            .limit(top_k)
            .to_df()
        )
        return results.to_string()


class LanceDBVectorSearch(BaseRetriever):
    """LangChain-compatible vector search retriever backed by LanceDB.

    Attributes:
        table: The LanceDB table to search.
        embeddings: The embeddings model to use.
    """

    table: lancedb.table.LanceTable = Field(...)
    embeddings: HuggingFaceEmbeddings = Field(...)

    def __init__(self, table: lancedb.table.LanceTable, embeddings: HuggingFaceEmbeddings):
        super().__init__(table=table, embeddings=embeddings)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents relevant to the query via vector search."""
        query_embedding = np.array(self.embeddings.embed_query(query))

        results = (
            self.table.search(query_embedding, vector_column_name="embedding")
            .limit(2)
            .to_pandas()
        )

        return [
            Document(page_content=row["text"], metadata={})
            for _, row in results.iterrows()
        ]


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
            f"\n\n===== Document {i} =====\n" + doc.page_content
            for i, doc in enumerate(docs)
        )

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_agent(
    hybrid_search_tool: HybridSearchTool,
    model_id: str,
) -> CodeAgent:
    """Instantiate the SmolLM2 model and configure the CodeAgent."""
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
    # Load and chunk the PDF
    docs = load_and_chunk_pdf(PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    print(len(docs))

    # Build the embedding model and populate LanceDB
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    table = create_lancedb_table(LANCEDB_PATH, TABLE_NAME, docs, embeddings)

    # Build retrieval tools
    bm25_retriever = BM25Retriever.from_documents(docs, k=BM25_K)
    semantic_retriever = LanceDBVectorSearch(table=table, embeddings=embeddings)
    hybrid_search_tool = HybridSearchTool(bm25_retriever, semantic_retriever)

    # Quick smoke-test of hybrid search
    results = hybrid_search_tool.forward("Smart Home technologies", top_k=2)
    print(results)

    # Build and run the agent
    agent = build_agent(hybrid_search_tool, MODEL_ID)
    agent_output = agent.run(AGENT_QUERY)
    print("Final output:")
    print(agent_output)


if __name__ == "__main__":
    main()
