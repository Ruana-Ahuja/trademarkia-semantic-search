# Semantic Search System — 20 Newsgroups

Built a Semantic Search Engine that takes the 20 Newsgroups dataset (~20k Usenet posts), embeds them, clusters them using Fuzzy C-Means, and exposes a semantic search API with a cache that recognises paraphrased queries.

## Project Structure
```
trademarkia_semantic_search/
├── main.py              
├── data_loader.py       
├── embeddings.py        
├── clusterings.py       
├── semantic_cache.py    
├── requirements.txt     
└── data/                
```

## Getting Started

Create and activate a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Place the 20 Newsgroups dataset inside a folder called `data/` at the root of the project. Each category should be a subfolder with numbered files inside it.

Start the server:
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

First run takes around 5 minutes to embed and cluster everything. After that it loads from disk and starts in about 3 seconds.

Open `http://127.0.0.1:8000/docs` to test the endpoints in the browser.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /query | Semantic search with cache |
| GET | /cache/stats | Cache statistics |
| DELETE | /cache | Flush cache |

## Example Outputs

Cache miss on first query:
```
{
  "query": "space shuttle launch",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0,
  "result": "[1] (score=0.4097)\nIn article (Greg Moore) writes: As Henry pointed out, you have to develop the thruster...\n\n[2] (score=0.3721)\nWe are not at the end of the Space Age, but only at the end of its beginning...\n\n[3] (score=0.3533)\nWhile we are on the subject of the shuttle software...",
  "dominant_cluster": 2
}
```

Cache hit on a paraphrased query:
```
{
  "query": "latest hockey scores",
  "cache_hit": true,
  "matched_query": "hockey game results",
  "similarity_score": 0.8127,
  "result": "[1] (score=0.6603)\nHere are the standings after game 2 of each of the divisional semi-final series...",
  "dominant_cluster": 0
}
```

Cache stats:
```
{
  "total_entries": 10,
  "hit_count": 4,
  "miss_count": 6,
  "hit_rate": 0.4
}
```

## How It Works

Raw Usenet posts contain a lot of noise, such as quoted reply lines, email headers, signatures, URLs. These get stripped before embedding. Posts under 20 words after cleaning are dropped since they carry no useful content.

Embeddings use `all-MiniLM-L6-v2`. Fast on CPU, 384 dimensions, works well for retrieval on this scale. Embeddings and the clusterer are saved to disk after the first run so subsequent startups are fast.

Clustering uses Fuzzy C-Means. Unlike k-means, each document gets a probability distribution over all clusters rather than a single label. A post about gun control sits between politics and firearms, reflected by FCM. Cluster count is set to 15 rather than 20 because several newsgroup categories overlap in embedding space and silhouette analysis supported a lower number.

The semantic cache compares incoming query embeddings against cached ones using cosine similarity. A match above 0.80 returns the cached result without searching the documents again. At 0.70 the threshold is too loose and unrelated queries collide. At 0.90 it is too strict and paraphrases miss. 0.80 is where the system behaves correctly, such as the hockey example above with similarity 0.8127 is a concrete case of it working as intended.
