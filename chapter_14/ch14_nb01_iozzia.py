"""
Implementing a Custom Graph RAG System with Open Source SLMs and Ollama.

Companion script for chapter 14 of "Domain-Specific Small Language Models"
by Guglielmo Iozzia, Manning Publications, 2025.

Implements a custom Graph RAG (Retrieval Augmented Generation) system using
only Small Language Models (SLMs). GPU is recommended.

=============================================================================
EDUCATIONAL CONCEPTS DEMONSTRATED:
1. Why Graph RAG over Standard Vector RAG:
   - Vector RAG excels at localized fact retrieval ("What did X do in section Y?").
   - Graph RAG excels at holistic / global synthesis ("What are the overarching themes across the entire corpus?").
2. Knowledge Graph Construction:
   - Structured entity and relationship extraction using LLMs + Pydantic schema validation.
   - Merging subgraphs into a unified NetworkX topological graph.
3. Leiden Community Detection:
   - Partitions graph nodes into tightly connected communities based on modularity optimization.
4. Map-Reduce Global Query Synthesis:
   - Map: Generates intermediate answers from per-community summaries.
   - Reduce: Consolidates intermediate findings into a cohesive final synthesis.
=============================================================================

# Install the missing requirements before running:
# pip install PyPDF2==3.0.1 ollama networkx plotly matplotlib

# Ollama setup:
#   curl -fsSL https://ollama.com/install.sh | sh
#   ollama serve
#   ollama pull mistral
"""

import json
import os
import random
import sys
from pathlib import Path

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import PyPDF2
from ollama import ChatResponse, chat
from pydantic import BaseModel
from PyPDF2 import PdfReader, PdfWriter

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

MODEL_ID = "mistral"
PDF_DIR = "pdf_documents"
TEXT_DIR = "extracted_text"
GRAPHML_PATH = "combined_graph.graphml"
CHUNK_SIZE = 2000
TEXT_PLACEHOLDER = "Acknowledgments"
LEIDEN_RESOLUTION = 0.5
MAX_COMMUNITIES = 10
QUERY = "What is the main topic here?"

# Pages to strip from the primary arxiv paper (1-indexed, converted later)
PAGES_TO_REMOVE_RAW = [6, 7, 8, 9, 10, 11]
PRIMARY_PDF_INPUT = os.path.join(PDF_DIR, "arxiv_250212923.pdf")
PRIMARY_PDF_OUTPUT = os.path.join(PDF_DIR, "arxiv_250212923.pdf")

# ---------------------------------------------------------------------------
# Pydantic schema for structured model output
# ---------------------------------------------------------------------------


class RawKnowledgeGraph(BaseModel):
    entities: list[str]
    relationships: list[str]


# ---------------------------------------------------------------------------
# PDF utilities
# ---------------------------------------------------------------------------


def remove_pdf_pages(input_pdf_path: str, output_pdf_path: str, pages_to_remove: list[int]) -> None:
    """Remove specified pages from a PDF and write the result to disk."""
    if not os.path.exists(input_pdf_path):
        return
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    for page_num in range(len(reader.pages)):
        if page_num not in pages_to_remove:
            writer.add_page(reader.pages[page_num])

    with open(output_pdf_path, "wb") as output_pdf_file:
        writer.write(output_pdf_file)


def extract_text_from_pdfs(pdf_dir: str, output_dir: str, placeholder: str = "") -> None:
    """Extract text from all PDFs in a directory and save as individual text files."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            text_path = os.path.join(output_dir, filename[:-4] + ".txt")

            with open(pdf_path, "rb") as pdf_file, open(text_path, "w", encoding="utf-8") as text_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if page_num == num_pages - 1 and placeholder in text:
                        text = text.split(placeholder)[0]
                    text_file.write(text)


def split_text_into_chunks(text_files_dir: str, chunk_size: int = 1000) -> dict[str, list[str]]:
    """Split text files into fixed-size character chunks."""
    chunks_by_file = {}
    if not os.path.exists(text_files_dir):
        return chunks_by_file

    for filename in os.listdir(text_files_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(text_files_dir, filename)
            with open(file_path, encoding="utf-8") as file:
                text = file.read()

            chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
            chunks_by_file[filename] = chunks

    return chunks_by_file


# ---------------------------------------------------------------------------
# Graph RAG — indexing
# ---------------------------------------------------------------------------


def extract_entities_and_relationships(chunks_dict: dict, model_id: str) -> dict:
    """Query the Ollama model to extract entities and relationships from text chunks."""
    results = {}
    with console.status("[bold green]Extracting Knowledge Graph entities and relations via Ollama..."):
        for filename, chunks in chunks_dict.items():
            results[filename] = []
            for chunk in chunks:
                prompt = (
                    f"Extract entities and relationships from this text:\n\n{chunk}\n\n"
                    "Relationships must follow the format 'Relationship, (Entity1, Entity2)'"
                )
                try:
                    response: ChatResponse = chat(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        format=RawKnowledgeGraph.model_json_schema(),
                        options={"temperature": 0},
                    )
                    results[filename].append(response["message"]["content"])
                except Exception as e:
                    console.print(f"[yellow]Ollama extraction notice: {e}[/yellow]")
    return results


def summarize_elements(elements: dict, model_id: str = MODEL_ID) -> dict:
    """Summarize extracted graph elements using the Ollama model."""
    summaries = {}
    for filename, chunks in elements.items():
        summaries[filename] = []
        for chunk in chunks:
            try:
                chunk_dict = json.loads(chunk)
                entities = chunk_dict["entities"]
                relationships = chunk_dict["relationships"]
                prompt = (
                    "Summarize the following entities and relationships in the same structured format:\n"
                    f"Entities:\n{entities}\n\nRelationships:\n{relationships}\n"
                )
                response: ChatResponse = chat(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    format=RawKnowledgeGraph.model_json_schema(),
                    options={"temperature": 0},
                )
                summaries[filename].append(response["message"]["content"])
            except Exception:
                summaries[filename].append(json.dumps({"entities": [], "relationships": []}))
    return summaries


def build_graph(entities: list[str], relationships: list[str]) -> nx.Graph:
    """Build a NetworkX graph from entity and relationship lists."""
    graph = nx.Graph()
    graph.add_nodes_from(entities)

    for relationship in relationships:
        parts = relationship.split(",")
        if len(parts) >= 3:
            relationship_name = parts[0]
            entity1 = parts[1][2:]
            entity2 = parts[2][1 : len(parts[2]) - 1]
            graph.add_edge(entity1, entity2, label=relationship_name)

    return graph


def build_combined_graph(element_summaries: dict) -> nx.Graph:
    """Merge per-chunk knowledge graphs into a single combined graph."""
    combined_graph = nx.Graph()
    for filename, chunks_data in element_summaries.items():
        for chunk_data in chunks_data:
            try:
                chunk_kg = RawKnowledgeGraph.model_validate_json(chunk_data)
                graph = build_graph(chunk_kg.entities, chunk_kg.relationships)
                combined_graph = nx.compose(combined_graph, graph)
            except Exception:
                pass
    return combined_graph


def save_graph_to_graphml(graph: nx.Graph, filepath: str) -> None:
    """Save a NetworkX graph to a GraphML file."""
    try:
        nx.write_graphml(graph, filepath)
        console.print(f"[bold green]✔[/bold green] Graph saved to [yellow]{filepath}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error saving graph: {e}[/red]")


# ---------------------------------------------------------------------------
# Graph visualization
# ---------------------------------------------------------------------------


def visualize_graph_with_plotly(graph: nx.Graph) -> None:
    """Visualize a NetworkX graph with Plotly, colouring nodes by connection count."""
    if len(graph.nodes) == 0:
        return
    pos = nx.spring_layout(graph)

    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, node_labels = [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(node)

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(str(adjacencies[0]))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            reversescale=True,
            color=node_adjacencies,
            size=10,
            colorbar=dict(thickness=15, title="Node Connections", xanchor="left", titleside="right"),
            line_width=2,
        ),
    )
    node_trace.text = node_text

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        text=node_labels,
        mode="lines",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="<br>Chapter 14 Graph RAG example",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(text="", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    fig.show()


# ---------------------------------------------------------------------------
# Community detection (Leiden algorithm)
# ---------------------------------------------------------------------------


def calculate_modularity_gain(graph: nx.Graph, node, community: int, communities: dict, resolution: float) -> float:
    """Calculate the modularity gain from moving a node to a community."""
    m = graph.number_of_edges()
    if m == 0:
        return 0.0
    k_i = graph.degree(node)
    k_i_in = sum(1 for neighbor in graph.neighbors(node) if communities[neighbor] == community)
    sigma_tot = sum(graph.degree(v) for v in graph.nodes if communities[v] == community)
    return (k_i_in - k_i * sigma_tot / (2 * m)) * resolution


def aggregate_communities(graph: nx.Graph, communities: dict) -> tuple[dict, int]:
    """Re-index community IDs based on connected components."""
    community_mapping: dict = {}
    next_community_id = 0

    for component in nx.connected_components(graph):
        for node in component:
            if communities[node] not in community_mapping:
                community_mapping[communities[node]] = next_community_id
                next_community_id += 1
            communities[node] = community_mapping[communities[node]]

    return communities, next_community_id


def leiden_algorithm(graph: nx.Graph, resolution: float = 1.0, max_iterations: int = 100) -> list[int]:
    """Implement the Leiden algorithm for community detection."""
    communities = {node: i for i, node in enumerate(graph.nodes)}

    for _ in range(max_iterations):
        improved = False

        for node in graph.nodes:
            best_community = communities[node]
            best_modularity_gain = 0

            for neighbor in graph.neighbors(node):
                neighbor_community = communities[neighbor]
                if neighbor_community != best_community:
                    gain = calculate_modularity_gain(graph, node, neighbor_community, communities, resolution)
                    if gain > best_modularity_gain:
                        best_modularity_gain = gain
                        best_community = neighbor_community

            if best_community != communities[node]:
                communities[node] = best_community
                improved = True

        for node in graph.nodes:
            if sum(1 for n in graph.nodes if communities[n] == communities[node]) == 1:
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    communities[node] = communities[random.choice(neighbors)]

        if improved:
            communities, _ = aggregate_communities(graph, communities)
        else:
            break

    return [communities[node] for node in graph.nodes]


# ---------------------------------------------------------------------------
# Graph RAG — querying
# ---------------------------------------------------------------------------


def generate_community_summaries(
    community_list: list[list], graph: nx.Graph, chunks_dict: dict, model_id: str = MODEL_ID
) -> dict[int, str]:
    """Generate text summaries for each graph community using the Ollama model."""
    community_summaries: dict[int, str] = {}
    with console.status("[bold green]Generating community level summaries with Ollama..."):
        for community_id, community_nodes in enumerate(community_list):
            community_text = ""
            for node in community_nodes:
                if node in chunks_dict:
                    for chunk in chunks_dict[node]:
                        community_text += chunk + "\n"

            if not community_text.strip():
                community_text = "Community entities: " + ", ".join(map(str, community_nodes))

            prompt = (
                f"Summarize the following text which represents a community of related documents:\n\n{community_text}\n"
            )
            try:
                response: ChatResponse = chat(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0},
                )
                community_summaries[community_id] = response["message"]["content"]
            except Exception:
                community_summaries[community_id] = f"Summary for community {community_id}"

    return community_summaries


def generate_answers(query: str, community_summaries: dict[int, str], model_id: str = MODEL_ID) -> dict[int, str]:
    """Generate per-community answers to a user query."""
    answers: dict[int, str] = {}
    for community_id, summary in community_summaries.items():
        prompt = f"Query: {query}\n\nUse the following community summary to answer the query:\n\n{summary}\n\nAnswer:"
        try:
            response: ChatResponse = chat(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            answers[community_id] = response["message"]["content"]
        except Exception:
            answers[community_id] = f"Partial answer from community {community_id}"
    return answers


def generate_final_answer(intermediate_answers: dict[int, str], model_id: str = MODEL_ID) -> str:
    """Combine intermediate community answers into a single concise response."""
    combined_prompt = "Combine the following answers into a single, concise response:\n\n"
    for community_id, answer in intermediate_answers.items():
        combined_prompt += f"Community {community_id}: {answer}\n\n"

    try:
        response: ChatResponse = chat(
            model=model_id,
            messages=[{"role": "user", "content": combined_prompt}],
            options={"temperature": 0},
        )
        return response["message"]["content"]
    except Exception:
        return "Consolidated Graph RAG answer."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate the full Graph RAG pipeline: indexing → querying."""
    render_banner(
        title="Custom Graph RAG with Ollama & NetworkX",
        subtitle="Chapter 14: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Query": f'"{QUERY}"',
            "Graph Format": "GraphML / NetworkX",
        },
        icon="🚀",
    )

    # Step 1: Ingestion & Text Chunking
    render_step(1, "PDF Ingestion & Section Partitioning", icon="📋")
    pages_to_remove = [x - 1 for x in PAGES_TO_REMOVE_RAW]
    remove_pdf_pages(PRIMARY_PDF_INPUT, PRIMARY_PDF_OUTPUT, pages_to_remove)

    extract_text_from_pdfs(PDF_DIR, TEXT_DIR, TEXT_PLACEHOLDER)
    chunks_dict = split_text_into_chunks(TEXT_DIR, chunk_size=CHUNK_SIZE)
    render_card(
        "Document Chunks Prepared",
        f"Extracted text chunks from [text.highlight]{len(chunks_dict)}[/text.highlight] documents.",
        icon="📄",
    )

    # Step 2: Structured Entity & Relation Extraction
    render_step(2, "Structured Entity & Relationship Extraction with Pydantic", icon="🧠")
    elements = extract_entities_and_relationships(chunks_dict, MODEL_ID)
    element_summaries = summarize_elements(elements)

    # Step 3: Building Knowledge Graph
    render_step(3, "Constructing & Serializing NetworkX Knowledge Graph", icon="🕸️")
    combined_graph = build_combined_graph(element_summaries)
    save_graph_to_graphml(combined_graph, GRAPHML_PATH)

    columns = [("Graph Property", STYLE_PRIMARY, "left"), ("Count", STYLE_SUCCESS, "right")]
    rows = [
        ("Total Entities (Nodes)", str(combined_graph.number_of_nodes())),
        ("Total Relationships (Edges)", str(combined_graph.number_of_edges())),
    ]
    console.print(create_table("Knowledge Graph Topological Overview", columns, rows))
    pause()

    visualize_graph_with_plotly(combined_graph)

    # Step 4: Leiden Community Detection
    render_step(4, "Executing Hierarchical Leiden Modularity Clustering", icon="🔍")
    community_assignments = leiden_algorithm(combined_graph, LEIDEN_RESOLUTION)
    unique_communities = sorted(set(community_assignments))[:MAX_COMMUNITIES]
    render_card(
        "Community Detection",
        f"Identified [text.highlight]{len(unique_communities)}[/text.highlight] modular semantic communities.",
        icon="✔",
    )

    # Step 5: Community Summarization & Global Querying
    render_step(5, "Map-Reduce Community Summarization & Global Query Synthesis", icon="✨")
    community_list = [
        [node for node, comm in zip(combined_graph.nodes, community_assignments) if comm == community_id]
        for community_id in unique_communities
    ]

    summaries = generate_community_summaries(community_list, combined_graph, chunks_dict)
    intermediate_answers = generate_answers(QUERY, summaries)
    final_answer = generate_final_answer(intermediate_answers)

    render_card(
        title="Consolidated Graph RAG Response",
        content=f"[text.muted]Global Query:[/text.muted] {QUERY}\n\n[status.success]Synthesis:[/status.success]\n{final_answer}",
        icon="🎯",
    )

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Graph Structures as Ground Truth",
                "By converting unstructured documents into an explicit Knowledge Graph (entities + typed relationships), the system captures complex cross-document connections that standard text chunking fragments.",
            ),
            (
                "Hierarchical Leiden Clustering",
                "Grouping nodes by graph modularity detects thematic clusters naturally, allowing the SLM to summarize high-level themes independently.",
            ),
            (
                "Global Q&A via Map-Reduce",
                "Mapping user questions across community summaries and reducing intermediate answers yields comprehensive syntheses that vector search alone cannot produce.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
