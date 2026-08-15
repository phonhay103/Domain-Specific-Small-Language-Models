"""
Implementing a RAG System with Open Source SLMs and LanceDB.

This script is a companion of chapter 13 of the "Small Domain Specific LLMs in Action"
book, author Guglielmo Iozzia, Manning Publications, 2025.

The code implements a basic RAG (Retrieval Augmented Generation) system using only Small
Language Models (SLMs) and the Open Source vector database LanceDB
(https://lancedb.github.io/lancedb/). Data preprocessing, embedding transformation, and
retrieval do not require hardware acceleration. The answer generation process can run
with or without hardware acceleration, but loading model weights to a GPU is recommended
for speed.

More details about the code can be found in the related book's chapter.

Install missing dependencies if needed:
    pip install PyMuPDF lancedb llama-cpp-python
"""

# stdlib
import os
import random
import re
import textwrap
from time import perf_counter as timer

# third-party
import fitz
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import requests
import torch
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util
from spacy.lang.en import English
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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
TOP_K_RESULTS = 5
RETRIEVAL_TOP_K = 2
LLM_N_CTX = 1024
EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

def download_pdf(url: str, path: str) -> None:
    """Download a PDF from *url* and save it to *path* if not already present."""
    if not os.path.exists(path):
        print("File doesn't exist, downloading it...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, "wb") as file:
                file.write(response.content)
            print(f"The file has been downloaded and saved as {path}")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
    else:
        print(f"File {path} exists.")


def text_formatter(text: str) -> str:
    """Normalise raw PDF text by collapsing newlines and stripping whitespace."""
    cleaned_text = text.replace("\n", " ").strip()
    # Add here any other extra text formatting
    return cleaned_text


def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Open a PDF file, read its text content page by page, and collect statistics.

    Returns a list of dicts with page number, character/word/sentence/token counts,
    and the extracted text for each page.
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({
            "page_number": page_number + 1,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4,  # 1 token ≈ 4 characters
            "text": text,
        })
    return pages_and_texts


def sentencise_pages(pages_and_texts: list[dict]) -> list[dict]:
    """Split each page's text into sentences using SpaCy and count them."""
    nlp = English()
    nlp.add_pipe("sentencizer")
    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])
    return pages_and_texts


def split_list(input_list: list, slice_size: int) -> list[list]:
    """Split *input_list* into sub-lists of length *slice_size*."""
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


def chunk_sentences(pages_and_texts: list[dict], chunk_size: int) -> list[dict]:
    """Group each page's sentences into fixed-size chunks."""
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])
    return pages_and_texts


def build_chunk_dicts(pages_and_texts: list[dict]) -> list[dict]:
    """Flatten page sentence chunks into individual dicts with statistics."""
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict: dict = {}
            chunk_dict["page_number"] = item["page_number"]

            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token ≈ 4 characters

            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks


def filter_short_chunks(pages_and_chunks: list[dict], min_tokens: int) -> list[dict]:
    """Remove chunks with too few tokens (they likely contain negligible information)."""
    df = pd.DataFrame(pages_and_chunks)
    print(df.describe().round(2))

    subset_df = df[df["chunk_token_count"] <= min_tokens]
    for row in subset_df.sample(min(5, len(subset_df)), replace=True).iterrows():
        print(
            f'Chunk token count: {row[1]["chunk_token_count"]} | '
            f'Text: {row[1]["sentence_chunk"]}'
        )

    return df[df["chunk_token_count"] > min_tokens].to_dict(orient="records")


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[dict]:
    """Encode each chunk's sentence text as a dense embedding vector."""
    for item in tqdm(chunks):
        item["embedding"] = model.encode(item["sentence_chunk"])
    return chunks


def save_embeddings(chunks: list[dict], path: str) -> None:
    """Persist chunks with embeddings to a CSV file."""
    df = pd.DataFrame(chunks)
    df["embedding_str"] = df["embedding"].apply(lambda x: np.array2string(x, separator=",")[1:-1])
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# LanceDB vector store
# ---------------------------------------------------------------------------

def build_lancedb_table(db_path: str, table_name: str, csv_path: str) -> lancedb.db.LanceTable:
    """
    Load embeddings from CSV, write them to a LanceDB table, and create a cosine index.

    Returns the populated LanceDB table.
    """
    db = lancedb.connect(db_path)

    df_load = pd.read_csv(csv_path)
    df_load["embedding_final"] = df_load["embedding_str"].apply(
        lambda x: np.fromstring(x, sep=",")
    )
    print(df_load.head())

    final_data = []
    for row in df_load.itertuples(index=False):
        temp = {
            "page_number": row.page_number,
            "chunk_char_count": row.chunk_char_count,
            "chunk_word_count": row.chunk_word_count,
            "chunk_token_count": row.chunk_token_count,
            "sentence_chunk": row.sentence_chunk,
            "embedding": np.array(row.embedding_final),
        }
        final_data.append(temp)

    data_dict: dict = {}
    for key in final_data[0].keys():
        data_dict[key] = [d[key] for d in final_data]

    table = db.create_table(
        table_name,
        data=pa.Table.from_pydict(data_dict),
        mode="overwrite",
        schema=pa.schema([
            ("page_number", pa.int64()),
            ("chunk_char_count", pa.int64()),
            ("chunk_word_count", pa.int64()),
            ("chunk_token_count", pa.float64()),
            ("sentence_chunk", pa.string()),
            ("embedding", pa.list_(pa.float32(), list_size=EMBEDDING_DIM)),
        ]),
    )

    db[table_name].create_index(
        metric="cosine",
        vector_column_name="embedding",
        index_type="IVF_FLAT",
    )
    return table


def search_lancedb(db_path: str, table_name: str, query_embedding: np.ndarray, limit: int = 3) -> list[dict]:
    """Query the LanceDB table using a pre-computed embedding and print results."""
    db = lancedb.connect(db_path)
    results = (
        db[table_name]
        .search(query_embedding, vector_column_name="embedding")
        .limit(limit)
        .to_list()
    )
    print("Search results:")
    for result in results:
        print(result["_distance"])
        print(result["sentence_chunk"])
        print(result["page_number"])
        print("")
    return results


# ---------------------------------------------------------------------------
# Retrieval helpers (embeddings on file)
# ---------------------------------------------------------------------------

def load_embeddings_from_csv(path: str, device: str) -> tuple[list[dict], torch.Tensor]:
    """
    Load chunk embeddings from a CSV file.

    Returns the list of chunk dicts and a GPU/CPU tensor of all embeddings.
    """
    df = pd.read_csv(path)
    # Embedding column was serialised to string when saved
    df["embedding"] = df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ")
    )
    pages_and_chunks = df.to_dict(orient="records")
    # NumPy arrays are float64; Torch tensors default to float32
    embeddings = torch.tensor(
        np.array(df["embedding"].tolist()), dtype=torch.float32
    ).to(device)
    return pages_and_chunks, embeddings


def retrieve_top_k_dot(
    query: str,
    embedding_model: SentenceTransformer,
    embeddings: torch.Tensor,
    pages_and_chunks: list[dict],
    k: int = TOP_K_RESULTS,
) -> None:
    """Retrieve the top-k most relevant chunks using dot-product similarity."""
    print(f"Query: {query}")
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    start = timer()
    dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
    elapsed = timer() - start
    print(f"Time taken to get scores on {len(embeddings)} embeddings: {elapsed:.5f} seconds.")

    top_results = torch.topk(dot_scores, k=k)
    print(top_results)

    print_results(query, top_results, pages_and_chunks)


def retrieve_top_k_cosine(
    query: str,
    embedding_model: SentenceTransformer,
    embeddings: torch.Tensor,
    pages_and_chunks: list[dict],
    k: int = TOP_K_RESULTS,
) -> torch.return_types.topk:
    """Retrieve the top-k most relevant chunks using cosine similarity."""
    print(f"Query: {query}")
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    start = timer()
    cosine_scores = util.cos_sim(a=query_embedding, b=embeddings)[0]
    elapsed = timer() - start
    print(f"Time taken to get scores on {len(embeddings)} embeddings: {elapsed:.5f} seconds.")

    top_results = torch.topk(cosine_scores, k=k)
    print(top_results)
    return top_results


def print_wrapped(text: str, wrap_length: int = 80) -> None:
    """Print *text* word-wrapped at *wrap_length* characters."""
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


def print_results(query: str, top_results, pages_and_chunks: list[dict]) -> None:
    """Display query results in descending relevance-score order."""
    print(f"Query: '{query}'\n")
    print("Results:")
    for score, idx in zip(top_results[0], top_results[1]):
        print(f"Score: {score:.4f}")
        print("Text:")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def load_llm(repo_id: str, filename: str, n_ctx: int) -> Llama:
    """Download and initialise the GGUF model from the HF Hub."""
    return Llama.from_pretrained(repo_id=repo_id, filename=filename, verbose=False, n_ctx=n_ctx)


def prompt_formatter(query: str, context_items: list[dict], tokenizer) -> str:
    """
    Build a RAG prompt by combining *query*, retrieved *context_items*, and a base template.

    Returns the fully formatted prompt string ready for the generator.
    """
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    base_prompt = (
        "Based on the following context items, please answer the query.\n"
        "Make sure your answers are as explanatory as possible.\n"
        "\nUse the following context items to answer the user query:\n"
        "{context}\n"
        "\nRelevant passages: <extract relevant passages from the context here>\n"
        "User query: {query}\n"
        "Answer:"
    )
    base_prompt = base_prompt.format(context=context, query=query)

    dialogue_template = [{"role": "user", "content": base_prompt}]
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def generate_answer(llm: Llama, prompt: str) -> None:
    """Run the GGUF model on *prompt* and print the output."""
    output = llm(
        prompt,
        max_tokens=None,
        stop=["Q:", "\n"],
        echo=True,
    )
    print(output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end RAG pipeline: preprocess → embed → retrieve → generate."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data Preprocessing ---
    download_pdf(PDF_URL, PDF_PATH)
    pages_and_texts = open_and_read_pdf(PDF_PATH)
    print(pages_and_texts[:2])

    pages_and_texts = sentencise_pages(pages_and_texts)
    pages_and_texts = chunk_sentences(pages_and_texts, NUM_SENTENCE_CHUNK_SIZE)
    pages_and_chunks = build_chunk_dicts(pages_and_texts)
    print(len(pages_and_chunks))

    pages_and_chunks_filtered = filter_short_chunks(pages_and_chunks, MIN_TOKEN_LENGTH)
    print(pages_and_chunks_filtered[:2])

    # Embed chunks with the Sentence Transformer
    embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME, device="cpu")
    pages_and_chunks_filtered = embed_chunks(pages_and_chunks_filtered, embedding_model)
    save_embeddings(pages_and_chunks_filtered, EMBEDDINGS_SAVE_PATH)

    # --- LanceDB vector store ---
    lancedb_table = build_lancedb_table(LANCEDB_PATH, LANCEDB_TABLE, EMBEDDINGS_SAVE_PATH)

    # Query LanceDB
    query_embedding_np = embedding_model.encode(QUERY, convert_to_tensor=False)
    lancedb_results = search_lancedb(LANCEDB_PATH, LANCEDB_TABLE, query_embedding_np, limit=3)

    # --- Search using embeddings on file ---
    pages_and_chunks_loaded, embeddings = load_embeddings_from_csv(EMBEDDINGS_SAVE_PATH, device)

    # Re-create the embedding model on the target device (in case we resume from saved embeddings)
    embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME, device=device)

    # Dot-product retrieval
    retrieve_top_k_dot(QUERY, embedding_model, embeddings, pages_and_chunks_loaded)

    # Cosine similarity retrieval
    top_cosine = retrieve_top_k_cosine(QUERY, embedding_model, embeddings, pages_and_chunks_loaded)

    # --- Generation ---
    llm = load_llm(GENERATIVE_MODEL_REPO, GENERATIVE_MODEL_FILE, LLM_N_CTX)
    gen_tokenizer = AutoTokenizer.from_pretrained(GENERATIVE_TOKENIZER_ID)

    # Retrieve top-2 context chunks via cosine similarity for RAG generation
    query_embedding_tensor = embedding_model.encode(QUERY, convert_to_tensor=True)
    cosine_scores = util.cos_sim(a=query_embedding_tensor, b=embeddings)[0]
    scores, indices = torch.topk(cosine_scores, k=RETRIEVAL_TOP_K)
    context_items = [pages_and_chunks_loaded[i] for i in indices]

    # Generate answer using embeddings-on-file retrieval
    prompt = prompt_formatter(query=QUERY, context_items=context_items, tokenizer=gen_tokenizer)
    print(prompt)
    generate_answer(llm, prompt)

    # Generate answer using LanceDB retrieval
    # (requires the LanceDB section above to have been executed)
    prompt_lancedb = prompt_formatter(
        query=QUERY, context_items=lancedb_results[0:2], tokenizer=gen_tokenizer
    )
    print(prompt_lancedb)
    generate_answer(llm, prompt_lancedb)


if __name__ == "__main__":
    main()
