import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIGURATION
# ==============================
FOLDER_PATH = "plant_docs"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_RESULTS = 3
SIMILARITY_THRESHOLD = 0.25   # Tune based on your dataset

# ==============================
# LOAD MODEL
# ==============================
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# ==============================
# LOAD DOCUMENTS
# ==============================
documents = []
metadata = []

print("Loading documents...")

for filename in os.listdir(FOLDER_PATH):
    file_path = os.path.join(FOLDER_PATH, filename)

    if not filename.endswith(".txt"):
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

        if text:  # skip empty files
            documents.append(text)
            metadata.append(filename)

if len(documents) == 0:
    raise ValueError("No documents found in folder!")

print(f"Loaded {len(documents)} documents.")

# ==============================
# CREATE EMBEDDINGS
# ==============================
print("Creating embeddings...")
embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# ==============================
# CREATE FAISS INDEX (COSINE)
# ==============================
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("Vector index ready.\n")

# ==============================
# QUERY LOOP
# ==============================
while True:
    query = input("\nAsk something (or type 'exit'): ")

    if query.lower() == "exit":
        break

    # Embed query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Normalize query
    faiss.normalize_L2(query_embedding)

    # Dynamic k (avoid FAISS repetition issue)
    k = min(10, len(documents))

    scores, indices = index.search(query_embedding, k=k)

    print("\nTop Results:\n")

    shown_files = set()
    results_found = 0

    # Always show best result (even if low score)
    best_idx = indices[0][0]
    best_score = scores[0][0]

    print("---------------")
    print(f"File: {metadata[best_idx]}")
    print(f"Similarity Score: {round(float(best_score), 3)}\n")
    print(documents[best_idx])

    shown_files.add(metadata[best_idx])
    results_found += 1

    # Show additional relevant results
    for i in range(1, len(indices[0])):
        idx = indices[0][i]
        score = scores[0][i]
        filename = metadata[idx]

        if score < SIMILARITY_THRESHOLD:
            continue

        if filename in shown_files:
            continue

        print("\n---------------")
        print(f"File: {filename}")
        print(f"Similarity Score: {round(float(score), 3)}\n")
        print(documents[idx])

        shown_files.add(filename)
        results_found += 1

        if results_found >= MAX_RESULTS:
            break

    if results_found == 0:
        print("No relevant documents found.")