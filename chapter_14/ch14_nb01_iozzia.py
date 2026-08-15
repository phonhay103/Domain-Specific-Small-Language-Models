# ==========================================
# Extracted from CH14_NB01_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Implementing a Custom Graph RAG System with Open Source SLMs and Ollama
# This notebook is a companion of chapter 14 of the "Domain-Specific Small Language Models" [book](https://www.manning.com/books/domain-specific-small-language-models), author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2025.  
# The code in this notebook is about implementing a custom Graph RAG (Retrieval Augmented Generation) system using only Small Language Models (SLMs). Hardware acceleration (GPU) is recommended.   
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing requirements in the Colab VM (only PyPDF2, Colab-xterm and Ollama Python not available by default).
# ------------------------------------------------------------

# !pip install PyPDF2==3.0.1 colab-xterm ollama

# ------------------------------------------------------------
# ### Ollama setup
# Enable the Colab-xterm extension and start a terminal. Then install the Ollama server and start it by running the two commands below from the terminal:
# ```
# curl -fsSL https://ollama.com/install.sh | sh
# 
# ollama serve
# ```
# 
# ------------------------------------------------------------

# %load_ext colabxterm

# %xterm

# ------------------------------------------------------------
# Finally, pull the Mistal 7B model from the Ollama model registry.
# ------------------------------------------------------------

# !ollama pull mistral

# ------------------------------------------------------------
# ### Indexing
# ------------------------------------------------------------

# ------------------------------------------------------------
# Upload some PDF documents.
# ------------------------------------------------------------

# !mkdir pdf_documents
# %cd pdf_documents
# !curl https://arxiv.org/pdf/2502.12923 --output arxiv_250212923.pdf
#!curl https://arxiv.org/pdf/2502.12346 --output arxiv_250212346.pdf
#!curl https://arxiv.org/pdf/2407.11534 --output arxiv_240711534.pdf
# %cd ..

# ------------------------------------------------------------
# Let's do some data cleanup. Define a custom function to remove references and appendices from the uploaded PDF.
# ------------------------------------------------------------

from PyPDF2 import PdfReader, PdfWriter

def remove_pdf_pages(input_pdf_path, output_pdf_path, pages_to_remove):
    # Open the original PDF
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    # Iterate through the original PDF pages
    for page_num in range(len(reader.pages)):
        # Add pages that are not in the pages_to_remove list
        if page_num not in pages_to_remove:
            writer.add_page(reader.pages[page_num])

    # Write the new PDF to a file
    with open(output_pdf_path, 'wb') as output_pdf_file:
        writer.write(output_pdf_file)

# ------------------------------------------------------------
# Apply the `remove_pdf_pages` function to the uploaded PDF.
# ------------------------------------------------------------

input_pdf_path = "/content/pdf_documents/arxiv_250212923.pdf"
output_pdf_path = "/content/pdf_documents/arxiv_250212923.pdf"
pages_to_remove = [6, 7, 8, 9, 10, 11]
pages_to_remove = [x - 1 for x in pages_to_remove]
remove_pdf_pages(input_pdf_path, output_pdf_path, pages_to_remove)

# ------------------------------------------------------------
# Define a custom function to extract text from uploaded PDF documents.
# ------------------------------------------------------------

import os
import PyPDF2

def extract_text_from_pdfs(pdf_dir, output_dir, placeholder=''):
    """
    Extracts text from all PDF documents in a directory and saves them
    as individual text files.

    Args:
        pdf_dir: The path to the directory containing PDF documents.
        output_dir: The path to the directory where text files will be saved.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the PDF directory
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            text_path = os.path.join(output_dir, filename[:-4] + ".txt")  # Remove .pdf extension

            # Extract text from the PDF
            with open(pdf_path, "rb") as pdf_file, open(text_path, "w", encoding="utf-8") as text_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if page_num == num_pages - 1:
                        if placeholder in text:
                            text = text.split(placeholder)[0]
                    text_file.write(text)

    print("Text extraction completed.")

# ------------------------------------------------------------
# Apply the `extract_text_from_pdfs` function to the uploaded PDF.
# ------------------------------------------------------------

pdf_dir = "pdf_documents"
output_dir = "extracted_text"
placeholder = 'Acknowledgments'
extract_text_from_pdfs(pdf_dir, output_dir, placeholder)

# ------------------------------------------------------------
# Define a custom function to chunk the extracted text.
# ------------------------------------------------------------

def split_text_into_chunks(text_files_dir, chunk_size=1000):
    """
    Splits multiple text files into chunks of text.

    Args:
        text_files_dir: The path to the directory containing text files.
        chunk_size: The desired size of each chunk in characters (default: 1000).

    Returns:
        A dictionary where keys are filenames and values are lists of text chunks.
    """

    chunks_by_file = {}
    for filename in os.listdir(text_files_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(text_files_dir, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            chunks = []
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i + chunk_size])

            chunks_by_file[filename] = chunks

    return chunks_by_file

# ------------------------------------------------------------
# Chunk the extracted text.
# ------------------------------------------------------------

text_files_dir = "extracted_text"
chunks_dict = split_text_into_chunks(text_files_dir, chunk_size=2000)

# ------------------------------------------------------------
# Define the model for the model responses when extracting entities and relationships of a graph from the knowledge base.
# ------------------------------------------------------------

from pydantic import BaseModel

class RawKnowledgeGraph(BaseModel):
  entities: list[str]
  relationships: list[str]

# ------------------------------------------------------------
# Define a custom function to query the model hosted in the Ollama local server to extract entities and relationships from text chunks.
# ------------------------------------------------------------

from ollama import chat
from ollama import ChatResponse

def extract_entities_and_relationships(chunks_dict, model_id):
  results = {}
  for filename, chunks in chunks_dict.items():
        results[filename] = []
        for chunk in chunks:
            prompt = f"""Extract entities and relationships from this text:\n\n{chunk}\n\n
Relationships must follow the format 'Relationship, (Entity1, Entity2)'"""
            response: ChatResponse = chat(model=model_id, messages=[
              {
                'role': 'user',
                'content': prompt,
              },
            ],
            format=RawKnowledgeGraph.model_json_schema(),
            options={"temperature":0}
            )
            results[filename].append(response['message']['content'])

  return results

# ------------------------------------------------------------
# Extract entities and relationships from the knowledge base.
# ------------------------------------------------------------

elements = extract_entities_and_relationships(chunks_dict, 'mistral')

elements

# ------------------------------------------------------------
# Define a custom function to query the model hosted in the Ollama local server to summarize graph elements.
# ------------------------------------------------------------

import json

def summarize_elements(elements, model_id='mistral'):
    """
    Summarizes elements using the specified Ollama model.

    Args:
        elements: A dictionary where keys are filenames and values are lists of
                  dictionaries containing entities and relationships.
        model_id: The ID of the Ollama model to use for summarization.

    Returns:
        A dictionary with the same structure as the input 'elements', but with
        the values being summaries of the original entity/relationship lists.
    """
    summaries = {}
    for filename, chunks in elements.items():
        summaries[filename] = []
        for chunk in chunks:
            try:
                chunk_dict = json.loads(chunk)
                entities = chunk_dict['entities']
                relationships = chunk_dict['relationships']
                prompt = f"""Summarize the following entities and relationships in the same structured format:
Entities:
{entities}

Relationships:
{relationships}
                """
                response: ChatResponse = chat(model=model_id, messages=[
                    {'role': 'user', 'content': prompt}
                ],
                format=RawKnowledgeGraph.model_json_schema(),
                options={"temperature": 0}
                )
                summaries[filename].append(response['message']['content'])
            except Exception as e:
                print(f"Error processing chunk for {filename}: {e}")
                summaries[filename].append({"entities": [], "relationships": []})

    return summaries

# ------------------------------------------------------------
# Summarize the extracted graph elements.
# ------------------------------------------------------------

element_summaries = summarize_elements(elements)

element_summaries

# ------------------------------------------------------------
# Define a custom function to build a knowledge graph starting from entities and relationships identified by the SLM.
# ------------------------------------------------------------

import networkx as nx

def build_graph(entities, relationships):
  """
  Builds a graph from entities and relationships.

  Args:
    entities: A list of entities.
    relationships: A list of relationships in various formats.
                   Could be [('Relationship', 'Entity1', 'Entity2'), ...]
                   or other structures.

  Returns:
    A NetworkX graph object.
  """

  graph = nx.Graph()  # Create an undirected graph

  # Add nodes (entities)
  graph.add_nodes_from(entities)

  # Add edges (relationships), handling different relationship formats
  for relationship in relationships:
    relationship_as_list = relationship.split(',')
    # Assuming relationships are in the format ('Relationship', 'Entity1', 'Entity2')
    # If not, adjust the logic accordingly
    if len(relationship_as_list) >= 3:
      relationship_name = relationship_as_list[0]
      entity1 = relationship_as_list[1][2:]
      entity2 = relationship_as_list[2][1:len(relationship_as_list[2])-1]
      graph.add_edge(entity1, entity2, label=relationship_name)
      print(f"Adding relationship: {relationship}")
    else:
      # Handle cases where the relationship is not in the expected format
      print(f"Skipping relationship: {relationship} - Unexpected format")

  return graph

# ------------------------------------------------------------
# Cycle across the graph elements, build a knowledge graph for each chunk using the `build_graph` function and finally combine all the graph in a single one.
# ------------------------------------------------------------

combined_graph = nx.Graph()

for filename, chunks_data in element_summaries.items():
  for chunk_data in chunks_data:
    try:
      chunk_knowledge_graph = RawKnowledgeGraph.model_validate_json(chunk_data)
      entities = chunk_knowledge_graph.entities
      relationships = chunk_knowledge_graph.relationships

      graph = build_graph(entities, relationships)
      combined_graph = nx.compose(combined_graph, graph)

    except Exception as e:
      print(f"Error processing data for {filename}: {e}")

print("Combined graph built successfully!")

# ------------------------------------------------------------
# Define a custom function to visualize a graph with Plotly, with color node points by the number of connections.
# ------------------------------------------------------------

import plotly.graph_objects as go

def visualize_graph_with_plotly(graph):
  """
  Visualizes a NetworkX graph with Plotly, coloring node points
  by the number of connections.

  Args:
    graph: A NetworkX graph object.
  """

  pos = nx.spring_layout(graph)

  edge_x = []
  edge_y = []
  for edge in graph.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

  node_x = []
  node_y = []
  node_labels = []
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
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
      showscale=True,
      colorscale='YlGnBu',
      reversescale=True,
      color=[],
      size=10,
      colorbar=dict(
        thickness=15,
        title='Node Connections',
        xanchor='left',
        titleside='right'
      ),
      line_width=2))

  node_trace.marker.color = node_adjacencies
  node_trace.text = node_text

  edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    text=node_labels,
    mode='lines')

  fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
              title='<br>Chapter 13 Graph RAG example',
              titlefont_size=16,
              showlegend=False,
              hovermode='closest',
              margin=dict(b=20,l=5,r=5,t=40),
              annotations=[ dict(
                  text="",
                  showarrow=False,
                  xref="paper", yref="paper",
                  x=0.005, y=-0.002 ) ],
              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
              )
  fig.show()

# ------------------------------------------------------------
# Visualize the final knowledge graph.
# ------------------------------------------------------------

visualize_graph_with_plotly(combined_graph)

# ------------------------------------------------------------
# Define a custom function to save the combined graph to file in graphml format (to be easily exported and analyzed with other tools.
# ------------------------------------------------------------

import networkx as nx

def save_graph_to_graphml(graph, filepath):
  """Saves a NetworkX graph to a GraphML file.

  Args:
    graph: The NetworkX graph object to save.
    filepath: The path to the output GraphML file.
  """
  try:
    nx.write_graphml(graph, filepath)
    print(f"Graph saved to {filepath}")
  except Exception as e:
    print(f"Error saving graph: {e}")

save_graph_to_graphml(combined_graph, "combined_graph.graphml")

# ------------------------------------------------------------
# Define a custom function to implement the Leiden algorithm for community detection within a graph.
# ------------------------------------------------------------

import random

def leiden_algorithm(graph, resolution=1.0, max_iterations=100):
  """
  Implements the Leiden algorithm for community detection.

  Args:
    graph: A NetworkX graph object.
    resolution: The resolution parameter (default: 1.0).
    max_iterations: The maximum number of iterations (default: 100).

  Returns:
    A list of community assignments for each node.
  """
  communities = {node: i for i, node in enumerate(graph.nodes)}
  num_communities = len(communities)

  for _ in range(max_iterations):
    improved = False

    for node in graph.nodes:
      best_community = communities[node]
      best_modularity_gain = 0

      for neighbor in graph.neighbors(node):
        neighbor_community = communities[neighbor]
        if neighbor_community != best_community:
          modularity_gain = calculate_modularity_gain(graph, node, neighbor_community, communities, resolution)
          if modularity_gain > best_modularity_gain:
            best_modularity_gain = modularity_gain
            best_community = neighbor_community

      if best_community != communities[node]:
        communities[node] = best_community
        improved = True

    # Reassign singleton communities to a neighbor
    for node in graph.nodes:
      if len([n for n in graph.nodes if communities[n] == communities[node]]) == 1:
        neighbors = list(graph.neighbors(node))
        if neighbors:
          communities[node] = communities[random.choice(neighbors)]

    if improved:
      communities, num_communities = aggregate_communities(graph, communities)
    else:
      break

  return [communities[node] for node in graph.nodes]

# ------------------------------------------------------------
# Define a custom functio to calculate the modularity gain from moving a node to a community.
# ------------------------------------------------------------

def calculate_modularity_gain(graph, node, community, communities, resolution):
  """
  Calculates the modularity gain from moving a node to a community.
  """
  m = graph.number_of_edges()
  k_i = graph.degree(node)
  k_i_in = sum(1 for neighbor in graph.neighbors(node) if communities[neighbor] == community)
  sigma_tot = sum(graph.degree(v) for v in graph.nodes if communities[v] == community)

  return (k_i_in - k_i * sigma_tot / (2 * m)) * resolution

# ------------------------------------------------------------
# Define a custom function to aggregate communities based on modularity optimization.
# ------------------------------------------------------------

def aggregate_communities(graph, communities):
  """
  Aggregates communities based on modularity optimization.
  """

  # Create a mapping from original community IDs to new community IDs
  community_mapping = {}
  next_community_id = 0

  # Assign new community IDs based on connected components
  for component in nx.connected_components(graph):
    for node in component:
      if communities[node] not in community_mapping:
        community_mapping[communities[node]] = next_community_id
        next_community_id += 1

      # Update community assignment for the node
      communities[node] = community_mapping[communities[node]]

  num_communities = next_community_id

  return communities, num_communities

# ------------------------------------------------------------
# Use the Leiden algorithm to identify our graph communities.
# ------------------------------------------------------------

community_assignments = leiden_algorithm(combined_graph, 0.5)

for node, community in zip(combined_graph.nodes, community_assignments):
  print(f"Node {node} belongs to community {community}")

# ------------------------------------------------------------
# Plot the identified communities.
# ------------------------------------------------------------

import matplotlib.pyplot as plt

node_list = list(combined_graph.nodes)

node_colors = [community_assignments[node_list.index(node)] for node in combined_graph.nodes]

nx.draw(combined_graph, with_labels=True, node_color=node_colors)
plt.show()

# ------------------------------------------------------------
# Define a custom function that, starting from the community list and the generated graph, uses the Mistral model hosted in the local Ollama server to generate community summaries.
# ------------------------------------------------------------

def generate_community_summaries(community_list, graph, model_id="mistral"):
    """
    Generates summaries for each community in the graph using the Mistral model.

    Args:
        community_list: A list of communities, where each community is a list of node IDs.
        graph: The NetworkX graph representing the knowledge graph.
        model_id: The ID of the LLM model to use (default: "mistral").

    Returns:
        A dictionary where keys are community IDs and values are their summaries.
    """

    community_summaries = {}
    for community_id, community_nodes in enumerate(community_list):
        # Extract text chunks associated with nodes in the current community
        community_text = ""
        for node in community_nodes:
          # Assuming nodes correspond to filenames in chunks_dict
          if node in chunks_dict:
              for chunk in chunks_dict[node]:
                  community_text += chunk + "\n"

        # Generate summary using the Mistral model
        prompt = f"""Summarize the following text which represents a community of related documents:

{community_text}
        """
        response: ChatResponse = chat(model=model_id, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ], options={"temperature": 0})
        summary = response['message']['content']
        community_summaries[community_id] = summary

    return community_summaries

# ------------------------------------------------------------
# Get a list of the unique community assignments.
# ------------------------------------------------------------

unique_communities = sorted(list(set(community_assignments)))

unique_communities = unique_communities[:10]

# ------------------------------------------------------------
# Generate the community summaries for our graph.
# ------------------------------------------------------------

# Group nodes by community
community_list = []
for community_id in unique_communities:
    community_nodes = [node for node, community in zip(graph.nodes, community_assignments) if community == community_id]
    community_list.append(community_nodes)


summaries = generate_community_summaries(community_list, combined_graph)
summaries

# ------------------------------------------------------------
# ### Querying
# ------------------------------------------------------------

# ------------------------------------------------------------
# Define a custom function that generate answers to a given query from the community summaries returned by the `generate_community_summaries` function. It uses the local Ollama hosted SLM.
# ------------------------------------------------------------

def generate_answers(query, community_summaries, model_id="mistral"):
    """
    Generates answers to a given query based on community summaries.

    Args:
        query: The user's query.
        community_summaries: A dictionary of community summaries.
        model_id: The ID of the LLM model to use (default: "mistral").

    Returns:
        A dictionary where keys are community IDs and values are the answers
        generated for the query based on the corresponding community summary.
    """
    answers = {}
    for community_id, summary in community_summaries.items():
        prompt = f"""
        Query: {query}

        Use the following community summary to answer the query:

        {summary}

        Answer:
        """
        response: ChatResponse = chat(model=model_id, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ], options={"temperature": 0})
        answers[community_id] = response['message']['content']
    return answers

# ------------------------------------------------------------
# Generate the intermediate answers.
# ------------------------------------------------------------

query = 'What is the main topic here?'
intermediate_answers = generate_answers(query, summaries)

intermediate_answers

# ------------------------------------------------------------
# Define a custom function that generates the final answer from a user query, by combining the answers returned by `generate_answers function` into a final, concise response. It uses the local Ollama hosted SLM).
# ------------------------------------------------------------

def generate_final_answer(intermediate_answers, model_id="mistral"):
    """
    Combines intermediate answers into a final concise response.

    Args:
        intermediate_answers: A dictionary of intermediate answers from different communities.
        model_id: The ID of the LLM model to use (default: "mistral").

    Returns:
        A string representing the final answer.
    """

    # Combine intermediate answers into a single prompt
    combined_answers_prompt = "Combine the following answers into a single, concise response:\n\n"
    for community_id, answer in intermediate_answers.items():
        combined_answers_prompt += f"Community {community_id}: {answer}\n\n"

    # Generate the final answer using the LLM
    response: ChatResponse = chat(model=model_id, messages=[
        {
            'role': 'user',
            'content': combined_answers_prompt,
        },
    ], options={"temperature": 0})

    final_answer = response['message']['content']
    return final_answer

# ------------------------------------------------------------
# Generate the final answer to the given query.
# ------------------------------------------------------------

query = "What is the main topic here?"
final_answer = generate_final_answer(intermediate_answers)
final_answer

