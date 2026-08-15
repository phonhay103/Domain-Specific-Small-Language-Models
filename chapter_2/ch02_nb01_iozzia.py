# ==========================================
# Extracted from CH02_NB01_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # FAISS for TEXT - Quick Start
# This notebook is a companion of chapter 2 of the "Domain Specific LLms in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to introduce readers to the [FAISS](https://faiss.ai/index.html) library. No hardware acceleration required to execute all the code cells.  
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing required packages in the Colab VM. Only FAISS for CPU is missing. Then downgrade the now available [SentenceTransformers](https://www.sbert.net/) release to 4.1.0 for compatibility with the code in this notebook.
# ------------------------------------------------------------

# !pip install faiss-cpu
# !pip uninstall sentence-transformers
# !pip install sentence-transformers==4.1.0

# ------------------------------------------------------------
# Import the necessary packages/classes.
# ------------------------------------------------------------

"""Module to cluster embeddings and create indices."""
import faiss

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# Set the data corpus for this example and put it into a Pandas DataFrame.
# ------------------------------------------------------------

data = [['His secret identity is Peter Parker', 'spiderman'],
        ['A businessman and engineer who ' +
         'runs the company Stark Industries',
         'ironman'],
        ['Superhuman spider-powers and abilities ' +
         'after being bitten by a radioactive spider',
         'spiderman'],
        ['A frail man enhanced to the peak of human ' +
         'physical perfection by an experimental super-soldier serum', 'captainamerica']
        ]
df = pd.DataFrame(data, columns = ['text', 'context'])

# ------------------------------------------------------------
# Get embeddings from the data corpus, generate a FAISS index and add the embeddings to it (after normalization).  
# To make the code in the cell below compatible with SentenceTransformers release 5.0+, simple replace the line  
# ```vectors = encoder.encode(text)```  
# with  
# ```vectors = encoder.encode(text.to_list())```
# ------------------------------------------------------------

text = df['text']
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
vectors = encoder.encode(text)
vector_dimension = vectors.shape[1]
l2_index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
l2_index.add(vectors)

# ------------------------------------------------------------
# Prepare a search text to be used for similarity search with FAISS on the generated index.
# ------------------------------------------------------------

search_text = 'He throws webs'
search_vector = encoder.encode(search_text)
search_vector_as_array = np.array([search_vector])
faiss.normalize_L2(search_vector_as_array)

# ------------------------------------------------------------
# Perform a search within the created index (calculation of the distances between the search text and the strings within the index).
# ------------------------------------------------------------

k = l2_index.ntotal
distances, ann = l2_index.search(search_vector_as_array, k=k)

# ------------------------------------------------------------
# Prepare the results to be displayed in a user-friendly format.
# ------------------------------------------------------------

search_results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
merged_df = pd.merge(search_results, df, left_on='ann', right_index=True)
merged_df.head()

