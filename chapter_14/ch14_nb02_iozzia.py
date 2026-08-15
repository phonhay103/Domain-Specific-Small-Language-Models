# ==========================================
# Extracted from CH14_NB02_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Implementing an Agentic RAG System with Open Source SLMs and SmolAgents
# This notebook is a companion of chapter 14 of the "Domain-Specific Small Language Models" [book](https://www.manning.com/books/domain-specific-small-language-models), author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2025.  
# The code in this notebook is an example implementation of an Agentic RAG (Retrieval Augmented Generation) system using only Small Language Models (SLMs), the HF's [SmolAgents](https://github.com/huggingface/smolagents) framework, [LangChain](https://github.com/langchain-ai/langchain) and [LanceDB](https://github.com/lancedb/lancedb). Hardware acceleration (GPU) is recommended.   
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing requirements in the Colab VM.
# ------------------------------------------------------------

# !pip install smolagents langchain lancedb langchain-community rank_bm25 pypdf langchain-huggingface ddgs

# ------------------------------------------------------------
# Upload a PDF document.
# ------------------------------------------------------------

# !curl https://arxiv.org/pdf/2502.12923 --output arxiv_250212923.pdf

# ------------------------------------------------------------
# Extract the text from the uploaded document and chunk it.
# ------------------------------------------------------------

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("/content/arxiv_250212923.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

print(len(docs))

# ------------------------------------------------------------
# Download the embedding model, [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
# ------------------------------------------------------------

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ------------------------------------------------------------
# Create a local LanceDB database and the table where to store the embeddings. Then use the embedding model to transform the text chunks into embeddings and store them into the `document_embeddings` table.
# ------------------------------------------------------------

import lancedb
import numpy as np
import pyarrow as pa

db = lancedb.connect("./lancedb")

# Define the schema for the table
schema = pa.schema([
    pa.field("embedding", pa.list_(pa.float32(), list_size=384)), # Specify FixedSizeList for embedding
    pa.field("text", pa.string()),
])

table = db.create_table(
    "document_embeddings",
    schema=schema, # Pass the schema to create_table
    data=[
        {
            "embedding": np.array(embeddings.embed_query(doc.page_content), dtype=np.float32).flatten().tolist(),
            "text": doc.page_content,
        }
        for doc in docs
    ],
)

# ------------------------------------------------------------
# Define a custom SmolAgents tool to perform BM25 search on the knowledge base. BM25, or Best Matching 25, is a scoring algorithm used by search engines to evaluate how well a document matches a specific search query. It ranks documents based on factors like term frequency, document length, and the rarity of terms, making it effective for information retrieval tasks.
# ------------------------------------------------------------

from typing import List
from langchain.docstore.document import Document
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever

class BM25SearchTool(Tool):
    name = "do_bm25_search_on_local_documents"
    description = "Uses text search to retrieve the parts of the documentation that could be most relevant to answer a query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query string."
        },
    }
    output_type = "string"

    def __init__(self, docs: List[Document], **kwargs):
        super().__init__(**kwargs)
        self.docs = docs

    def forward(self, query: str) -> str:
        retriever = BM25Retriever.from_documents(self.docs, k=3)
        docs = retriever.invoke(query)
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {i} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )

# ------------------------------------------------------------
# Define a custom SmolAgents tool that performs semantic search on the `document_embeddings` table.
# ------------------------------------------------------------

from smolagents import Tool
import numpy as np

class SemanticSearchTool(Tool):
    name = "semantic_search"
    description = "Performs semantic search on a database of document embeddings."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query string.",
        },
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
        """Performs semantic search on the document_embeddings table."""
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array(query_embedding)
        results = self.table.search(query_embedding, vector_column_name="embedding").limit(top_k).to_df()
        return results.to_string()

# ------------------------------------------------------------
# Implement a custom SmolAgents Tool that does hybrid search (BM25 keyword search + vector search) on a LanceDB table. It uses LangChain's `EnsableRetriever` to combine the search results.
# ------------------------------------------------------------

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool

class HybridSearchTool(Tool):
    name = "hybrid_search"
    description = "Performs a hybrid search (BM25 keyword + vector search) on a LanceDB table."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query string.",
        },
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
            weights=[0.4, 0.6])

    def forward(self, query: str, top_k: int = 1) -> str:
        docs = self.ensemble_retriever.get_relevant_documents(query)
        # Process the results as needed
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {i} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )

# ------------------------------------------------------------
# Implement a LanceDB custom retriever class.
# ------------------------------------------------------------

from typing import List
from langchain.schema import BaseRetriever, Document
from pydantic import Field

class LanceDBVectorSearch(BaseRetriever):
    """
    A vector search retriever that uses LanceDB for storage and retrieval.

    Attributes:
        table (lancedb.LanceTable): The LanceDB table to search.
        embeddings (HuggingFaceEmbeddings): The embeddings model to use.
    """

    table: lancedb.table.LanceTable = Field(...) # Add Field for table
    embeddings: HuggingFaceEmbeddings = Field(...)

    def __init__(self, table: lancedb.table.LanceTable, embeddings: HuggingFaceEmbeddings):
        """
        Initializes the LanceDBVectorSearch retriever.

        Args:
            table (lancedb.LanceTable): The LanceDB table to search.
            embeddings (HuggingFaceEmbeddings): The embeddings model to use.
        """
        super().__init__(table=table, embeddings=embeddings) # Pass table and embeddings to super().__init__

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves relevant documents based on the given query.

        Args:
            query (str): The search query.

        Returns:
            List[Document]: A list of relevant documents.
        """
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array(query_embedding)

        # Perform the search in LanceDB
        results = self.table.search(query_embedding, vector_column_name="embedding").limit(2).to_pandas()  # Limit to top 10 results

        # Convert the results to Document objects
        documents = [
            Document(page_content=row["text"], metadata={})
            for _, row in results.iterrows()
        ]

        return documents

# ------------------------------------------------------------
# Test the hybrid search.
# ------------------------------------------------------------

bm25_retriever = BM25Retriever.from_documents(docs, k=3)
semantic_retriever = LanceDBVectorSearch(table=table, embeddings=embeddings)

hybrid_search_tool = HybridSearchTool(bm25_retriever, semantic_retriever)
results = hybrid_search_tool.forward("Smart Home technologies", top_k=2)
print(results)

# ------------------------------------------------------------
# Download the instructed SLM ([SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)) for the agent.
# ------------------------------------------------------------

import os
import torch
from smolagents import TransformersModel, CodeAgent, DuckDuckGoSearchTool

model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
model = TransformersModel(model_id,
                          device_map="auto",
                          max_new_tokens=700,
                          torch_dtype=torch.float16)

# ------------------------------------------------------------
# Create an instance of the SmolAgents' built-in `DuckDuckGoSearchTool`.
# ------------------------------------------------------------

from smolagents import DuckDuckGoSearchTool

duckduckgo_search_tool = DuckDuckGoSearchTool()

# ------------------------------------------------------------
# Create and setup the code agent, specifying the SLM to use and the list of allowed custom tools.
# ------------------------------------------------------------

custom_tools = [hybrid_search_tool, duckduckgo_search_tool]
agent = CodeAgent(
    tools=custom_tools,
    model=model,
    max_steps=3,
    verbosity_level=2,
    add_base_tools=False
)

# ------------------------------------------------------------
# Perform a query with the agent.
# ------------------------------------------------------------

#query = "Search the web for MSD Ireland"
#query = "Do an hybrid search about Smart Home technologies"
query = "Do an hybrid search about Smart Home technologies and then search the web about the same"
#query = "Do a semantic search for MSD"
agent_output = agent.run(query)
print("Final output:")
agent_output

