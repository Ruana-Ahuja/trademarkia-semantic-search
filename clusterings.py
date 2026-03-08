import numpy as np
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


class FuzzyClusterer:

    def __init__(self, n_clusters=15, m=2.0):
        self.n_clusters = n_clusters
        self.m = m
        self.centroids = None
        self.pca = None
        self.memberships = None

    def fit(self, embeddings):
        print("Training fuzzy clustering...")

        self.pca = PCA(n_components=64, random_state=42)
        reduced = self.pca.fit_transform(embeddings)
        reduced = normalize(reduced, norm="l2")

        data_T = reduced.T

        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            data=data_T,
            c=self.n_clusters,
            m=self.m,
            error=1e-5,
            maxiter=150,
            seed=42
        )

        self.centroids = cntr.astype(np.float32)
        self.memberships = u.T.astype(np.float32)

        print(f"FCM done. Fuzzy Partition Coefficient = {fpc:.4f}")

    def predict_cluster(self, embedding):
        probs = self.get_cluster_probabilities(embedding)
        return int(np.argmax(probs))

    def get_cluster_probabilities(self, embedding):
        reduced = self.pca.transform([embedding])
        reduced = normalize(reduced, norm="l2")[0]

        dists = np.linalg.norm(self.centroids - reduced, axis=1)

        if np.any(dists == 0):
            probs = np.zeros(self.n_clusters, dtype=np.float32)
            probs[np.argmin(dists)] = 1.0
            return probs

        exp = 2.0 / (self.m - 1.0)
        denom = np.sum((dists[:, None] / dists[None, :]) ** exp, axis=1)
        probs = (1.0 / denom).astype(np.float32)
        return probs