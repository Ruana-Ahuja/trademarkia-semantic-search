from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
import pickle

from data_loader import load_dataset
from embeddings import Embedder
from clusterings import FuzzyClusterer
from semantic_cache import SemanticCache


EMBEDDINGS_CACHE = "doc_embeddings.npy"
CLUSTERER_CACHE = "clusterer.pkl"

print("Loading dataset...")
documents = load_dataset("data")

if len(documents) == 0:
    raise ValueError("Dataset is empty. Check your data folder.")

print("Generating embeddings...")
embedder = Embedder()

if os.path.exists(EMBEDDINGS_CACHE):
    print("Loading embeddings from disk...")
    doc_embeddings = np.load(EMBEDDINGS_CACHE)
else:
    doc_embeddings = embedder.encode(documents)
    np.save(EMBEDDINGS_CACHE, doc_embeddings)
    print("Embeddings saved to disk.")

print("Embedding shape:", doc_embeddings.shape)

if os.path.exists(CLUSTERER_CACHE):
    print("Loading clusterer from disk...")
    with open(CLUSTERER_CACHE, "rb") as f:
        clusterer = pickle.load(f)
else:
    print("Building fuzzy clusters...")
    clusterer = FuzzyClusterer(n_clusters=15)
    clusterer.fit(doc_embeddings)
    with open(CLUSTERER_CACHE, "wb") as f:
        pickle.dump(clusterer, f)
    print("Clusterer saved to disk.")

doc_clusters = np.argmax(clusterer.memberships, axis=1)

print("Starting semantic cache...")
cache = SemanticCache(threshold=0.80)

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/")
def home():
    return {"message": "Semantic Search API running"}


@app.post("/query")
def query_endpoint(request: QueryRequest):
    query = request.query
    top_k = request.top_k

    query_embedding = embedder.encode([query])[0]
    cluster_probs = clusterer.get_cluster_probabilities(query_embedding)

    cached = cache.get(query_embedding)
    if cached is not None:
        result_string = "\n\n".join([
            f"[{i+1}] (score={r['score']})\n{r['text']}"
            for i, r in enumerate(cached["results"])
        ])
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cached["matched_query"],
            "similarity_score": cached["similarity_score"],
            "result": result_string,
            "dominant_cluster": int(np.argmax(cluster_probs))
        }

    top_clusters = np.argsort(cluster_probs)[::-1][:2]
    cluster_indices = np.where(np.isin(doc_clusters, top_clusters))[0]

    if len(cluster_indices) == 0:
        cluster_indices = np.arange(len(documents))

    cluster_embeddings = doc_embeddings[cluster_indices]
    similarities = cluster_embeddings @ query_embedding

    top_local = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_local:
        doc_index = cluster_indices[idx]
        results.append({
            "text": documents[doc_index][:300],
            "score": round(float(similarities[idx]), 4)
        })

    result_string = "\n\n".join([
        f"[{i+1}] (score={r['score']})\n{r['text']}"
        for i, r in enumerate(results)
    ])

    cache.add(query, query_embedding, results)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": 0.0,
        "result": result_string,
        "dominant_cluster": int(np.argmax(cluster_probs))
    }


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.delete("/cache")
def flush_cache():
    cache.flush()
    return {"message": "Cache flushed successfully."}