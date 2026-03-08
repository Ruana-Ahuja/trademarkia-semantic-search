import numpy as np


class SemanticCache:

    def __init__(self, threshold=0.80):
        self.threshold = threshold
        self.cache = []
        self.hit_count = 0
        self.miss_count = 0

    def get(self, query_embedding):
        if len(self.cache) == 0:
            self.miss_count += 1
            return None

        cached_embeddings = np.array([item["embedding"] for item in self.cache])
        similarities = cached_embeddings @ query_embedding

        best_index = similarities.argmax()
        best_score = float(similarities[best_index])

        if best_score >= self.threshold:
            self.hit_count += 1
            return {
                "results": self.cache[best_index]["results"],
                "matched_query": self.cache[best_index]["query"],
                "similarity_score": round(best_score, 4)
            }

        self.miss_count += 1
        return None

    def add(self, query_text, query_embedding, results):
        self.cache.append({
            "query": query_text,
            "embedding": query_embedding,
            "results": results
        })

    def stats(self):
        total = self.hit_count + self.miss_count
        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_count / total, 4) if total > 0 else 0.0
        }

    def flush(self):
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0