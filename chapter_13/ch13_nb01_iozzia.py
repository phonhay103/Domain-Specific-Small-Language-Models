# ==========================================
# Extracted from CH13_NB01_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Implementing a RAG System with Open Source SLMs and LanceDB
# This notebook is a companion of chapter 13 of the "Small Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2025.  
# The code in this notebook is about implementing a basic RAG (Retrieval Augmented Generation) system using only Small Language Models (SLMs) and a Open Source vector database, [LanceDB](https://lancedb.github.io/lancedb/). The data preprocessing, embedding transformation, retrieval with or without the vector database, don't require hardware acceleration. The answer generation process can run properly with or without hardware acceleration, but to make it faster, loading the model weights to a GPU is recommended.   
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing requirements (only PyMuPDF missing in the Colab VM).
# ------------------------------------------------------------

# !pip install PyMuPDF

# ------------------------------------------------------------
# ### Data Preprocessing
# ------------------------------------------------------------

# ------------------------------------------------------------
# Download the knowledge base (a single PDF document).
# ------------------------------------------------------------

import os
import requests

pdf_path = "2401.08671.pdf"

if not os.path.exists(pdf_path):
  print("File doesn't exist, downloading it...")

  url = "https://arxiv.org/pdf/2401.08671"

  filename = pdf_path

  response = requests.get(url)

  if response.status_code == 200:
      with open(filename, "wb") as file:
          file.write(response.content)
      print(f"The file has been downloaded and saved as {filename}")
  else:
      print(f"Failed to download the file. Status code: {response.status_code}")
else:
  print(f"File {pdf_path} exists.")

# ------------------------------------------------------------
# Implement a custom function that uses PyMuPDF to parse the PDF document to extract the raw text only (no tables, nor images) and get some statistics about it. A separate function for text formatting has been implemented too, so that extra ad hoc formatting can be easily added any time without touching the main parse function.
# ------------------------------------------------------------

import fitz
from tqdm.auto import tqdm

def text_formatter(text):
    cleaned_text = text.replace("\n", " ").strip()

    # Add here any other extra text formatting

    return cleaned_text

def open_and_read_pdf(pdf_path):
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number + 1,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,
                                "text": text})
    return pages_and_texts

# ------------------------------------------------------------
# Parse the PDF document.
# ------------------------------------------------------------

pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
pages_and_texts[:2]

# ------------------------------------------------------------
# Split the document's text into sentences using [SpaCy](https://spacy.io/). Force each sentence to string format, just in case, and calculate also the count of sentences per page.
# ------------------------------------------------------------

from spacy.lang.en import English

nlp = English()

nlp.add_pipe("sentencizer")

for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)

    item["sentences"] = [str(sentence) for sentence in item["sentences"]]

    item["page_sentence_count_spacy"] = len(item["sentences"])

# ------------------------------------------------------------
# Define a custom function to split a list into others of a given size. It will be used to split each page sentences into chunks.
# ------------------------------------------------------------

num_sentence_chunk_size = 10

def split_list(input_list, slice_size):

    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

# ------------------------------------------------------------
# Loop through pages and texts and split sentences into chunks.
# ------------------------------------------------------------

for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                         slice_size=num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])

# ------------------------------------------------------------
# Split each of the created chunks into its own item (ad get also some stats about each chunk).
# ------------------------------------------------------------

import re

pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

        pages_and_chunks.append(chunk_dict)

# ------------------------------------------------------------
# Calculate how many chucks we have.
# ------------------------------------------------------------

len(pages_and_chunks)

# ------------------------------------------------------------
# Optional step: filter sentences that have less than 30 tokens, as they would probably contain negligible information.
# ------------------------------------------------------------

import pandas as pd

df = pd.DataFrame(pages_and_chunks)
df.describe().round(2)

min_token_length = 30
subset_df = df[df["chunk_token_count"] <= min_token_length]

for row in subset_df.sample(min(5, len(subset_df)), replace=True).iterrows():
    print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')

pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
pages_and_chunks_over_min_token_len[:2]

# ------------------------------------------------------------
# Select an embedding model ([all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) in this example).
# ------------------------------------------------------------

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device="cpu")

# ------------------------------------------------------------
# Embed the chunks.
# ------------------------------------------------------------

for item in tqdm(pages_and_chunks_over_min_token_len):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])

# ------------------------------------------------------------
# Save the embeddings to file.
# ------------------------------------------------------------

import numpy as np

text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
text_chunks_and_embeddings_df['embedding_str'] = text_chunks_and_embeddings_df['embedding'].apply(
    lambda x: np.array2string(x, separator=',')[1:-1])
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

# ------------------------------------------------------------
# If you want to skip the vector database integration, you can jump straight to the [Search (Embeddings on File)](#scrollTo=7G8J_IWRJYKP) section.
# ------------------------------------------------------------

# ------------------------------------------------------------
# ### LanceDB
# ------------------------------------------------------------

# ------------------------------------------------------------
# This section introduced a vector database, LanceDB, to store the document embeddings.  
# Let's install the LanceDB package first.
# ------------------------------------------------------------

# !pip install lancedb

# ------------------------------------------------------------
# Create a local empty database instance and estabilish a connection to it.
# ------------------------------------------------------------

import lancedb

db = lancedb.connect("paperdb")

# ------------------------------------------------------------
# Load the preliminary saved embeddings from file and put them into a Pandas DataFrame.
# ------------------------------------------------------------

text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)
text_chunks_and_embedding_df_load['embedding_final'] = text_chunks_and_embedding_df_load['embedding_str'].apply(lambda x: np.fromstring(x, sep=','))
text_chunks_and_embedding_df_load.head()

# ------------------------------------------------------------
# Prepare embeddings and other data for ingestion to the database.
# ------------------------------------------------------------

final_data = []
for row in text_chunks_and_embedding_df_load.itertuples(index=False):
    temp = {}
    temp["page_number"] = row.page_number
    temp["chunk_char_count"] = row.chunk_char_count
    temp["chunk_word_count"] = row.chunk_word_count
    temp["chunk_token_count"] = row.chunk_token_count
    temp["sentence_chunk"] = row.sentence_chunk
    temp["embedding"] = np.array(row.embedding_final)
    final_data.append(temp)

# ------------------------------------------------------------
# Store embeddings and data in a LanceDB table ([PyArrow](https://arrow.apache.org/docs/python/index.html) format).
# ------------------------------------------------------------

import pyarrow as pa

data_dict = {}
for key in final_data[0].keys():
    data_dict[key] = [d[key] for d in final_data]

table = db.create_table(
    "paper_embeddings_table",
    data=pa.Table.from_pydict(data_dict),
    mode="overwrite",
    schema=pa.schema([
        ("page_number", pa.int64()),
        ("chunk_char_count", pa.int64()),
        ("chunk_word_count", pa.int64()),
        ("chunk_token_count", pa.float64()),
        ("sentence_chunk", pa.string()),
        ("embedding", pa.list_(pa.float32(), list_size=768)),
    ]),
)

# ------------------------------------------------------------
# Create an index over the table.
# ------------------------------------------------------------

db["paper_embeddings_table"].create_index(
    metric="cosine",
    vector_column_name="embedding",
    index_type="IVF_FLAT",
)

# ------------------------------------------------------------
# Provide a query and convert it into embeddings.
# ------------------------------------------------------------

query = "blocked KV-cache"
query_embedding = embedding_model.encode(query, convert_to_tensor=False)

# ------------------------------------------------------------
# Perform the search on the LanceDB table.
# ------------------------------------------------------------

results = db["paper_embeddings_table"].search(
        query_embedding,
        vector_column_name="embedding"
    ).limit(3).to_list()

# ------------------------------------------------------------
# Display the results.
# ------------------------------------------------------------

print("Search results:")
for result in results:
    print(result['_distance'])
    print(result['sentence_chunk'])
    print(result['page_number'])
    print("")

# ------------------------------------------------------------
# ### Search (Embeddings on File)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Load the embeddings preliminary saved on file and put them into a Pandas DataFrame.
# ------------------------------------------------------------

import random

import torch
import numpy as np
import pandas as pd

text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")

# ------------------------------------------------------------
# Convert the `embedding` column back to Numpy array (it was converted to string when saved to CSV).
# ------------------------------------------------------------

text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# ------------------------------------------------------------
# Convert the texts and embedding DataFrame to a list of dictionaries and then convert the embeddings to Torch tensors (numeric format conversion is required, as the NumPy arrays are Float 64, while Torch tensors are Float 32 by default).
# ------------------------------------------------------------

pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
embeddings.shape

# ------------------------------------------------------------
# Create again an instance of the Sentence Transformer, jsut in case we start the notebook on existing embeddings.
# ------------------------------------------------------------

from sentence_transformers import util, SentenceTransformer

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device=device)

# ------------------------------------------------------------
# Perform a query using the dot product algorithm. It returns the top 5 results.
# ------------------------------------------------------------

from sentence_transformers import util

query = "blocked KV-cache"
print(f"Query: {query}")

query_embedding = embedding_model.encode(query, convert_to_tensor=True)

from time import perf_counter as timer

start_time = timer()
dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
end_time = timer()

print(f"Time take to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

top_results_dot_product = torch.topk(dot_scores, k=5)
top_results_dot_product

# ------------------------------------------------------------
# Define a helper function to print wrapped text in the returned results in human readable format.
# ------------------------------------------------------------

import textwrap

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

# ------------------------------------------------------------
# Loop through the top result tuple and match up the scores and indices and then use them to index the `pages_and_chunks` variable to get the relevant text chunk. Results are printed to the cell output in descending relevance score order.
# ------------------------------------------------------------

print(f"Query: '{query}'\n")
print("Results:")
for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
    print(f"Score: {score:.4f}")
    print("Text:")
    print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
    print(f"Page number: {pages_and_chunks[idx]['page_number']}")
    print("\n")

# ------------------------------------------------------------
# Repeat the same query using cosine similarity.
# ------------------------------------------------------------

query = "blocked KV-cache"
print(f"Query: {query}")

query_embedding = embedding_model.encode(query, convert_to_tensor=True)

start_time = timer()
cosine_scores = util.cos_sim(a=query_embedding, b=embeddings)[0]
end_time = timer()

print(f"Time take to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

top_results_cosine_product = torch.topk(cosine_scores, k=5)
top_results_cosine_product

# ------------------------------------------------------------
# ### Generation
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the llama-cpp-python package.
# ------------------------------------------------------------

# !pip install llama-cpp-python

# ------------------------------------------------------------
# Download the [microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) model from the HF's Hub.
# ------------------------------------------------------------

from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="*-q4.gguf",
    verbose=False,
    n_ctx=1024
)

# ------------------------------------------------------------
# Provide a query and embed it.
# ------------------------------------------------------------

query = "blocked KV-cache"
query_embedding = embedding_model.encode(query, convert_to_tensor=True)

# ------------------------------------------------------------
# Do the retrieval using cosine similarity.
# ------------------------------------------------------------

cosine_scores = util.cos_sim(a=query_embedding, b=embeddings)[0]
scores, indices = torch.topk(cosine_scores, k=2)
context_items = [pages_and_chunks[i] for i in indices]

# ------------------------------------------------------------
# Download the tokenizer associated to the selected generative model.
# ------------------------------------------------------------

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# ------------------------------------------------------------
# Implement a custom function to format the prompt for the generator, by combining the query, the retrieved text chunks and a base prompt).
# ------------------------------------------------------------

def prompt_formatter(query,
                     context_items,
                     tokenizer):
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    base_prompt = """Based on the following context items, please answer the query.
Make sure your answers are as explanatory as possible.
\nUse the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    base_prompt = base_prompt.format(context=context, query=query)

    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt

# ------------------------------------------------------------
# Format the prompt.
# ------------------------------------------------------------

prompt = prompt_formatter(query=query,
                          context_items=context_items,
                          tokenizer=tokenizer)
prompt

# ------------------------------------------------------------
# Do the generation using the model in GGUF format.
# ------------------------------------------------------------

output = llm(
      prompt,
      max_tokens=None,
      stop=["Q:", "\n"],
      echo=True
)

# ------------------------------------------------------------
# Display the answer.
# ------------------------------------------------------------

output

# ------------------------------------------------------------
# The cells below require the local LanceDB database. Please go back to the [LanceDB section](#scrollTo=6RjcpGJimhv-) and execute the related code cells before moving further, if you haven't created and populated the database yet.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Format the query for retrieval from the database.
# ------------------------------------------------------------

prompt_lancedb = prompt_formatter(query=query,
                          context_items=results[0:2],
                          tokenizer=tokenizer)
prompt_lancedb

# ------------------------------------------------------------
# Do the generation using the model in GGUF format.
# ------------------------------------------------------------

output = llm(
      prompt_lancedb,
      max_tokens=None,
      stop=["Q:", "\n"],
      echo=True
)

# ------------------------------------------------------------
# Display the answer.
# ------------------------------------------------------------

output

