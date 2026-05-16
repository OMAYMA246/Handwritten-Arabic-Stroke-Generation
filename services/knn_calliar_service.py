import numpy as np
from sklearn.neighbors import NearestNeighbors

class KNNCalliarService:
    def __init__(self, npz_path='data/calliar_knn_data.npz'):
        data = np.load(npz_path)

        self.train_embs  = data['train_embs']   # (2494, 128) — déjà L2
        self.train_trajs = data['train_trajs']  # (2494, 128, 2)

        # KNN sur embeddings L2 — pas de z-score
        self.knn = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.knn.fit(self.train_embs)

        print(f'✅ KNN Calliar — {len(self.train_embs)} exemples')
        print(f'   Trajectoires : {self.train_trajs.shape}')

    def find_trajectory(self, embedding):
        # Normaliser L2 l'embedding utilisateur
        emb  = embedding.reshape(1, -1)
        emb  = emb / (np.linalg.norm(emb) + 1e-8)

        distances, indices = self.knn.kneighbors(emb, n_neighbors=3)

        # Moyenne pondérée des 3 voisins
        sims    = 1.0 - distances[0]
        sims    = np.clip(sims, 1e-6, None)
        weights = sims / sims.sum()

        trajectory = sum(
            w * self.train_trajs[indices[0][i]]
            for i, w in enumerate(weights)
        )

        distance   = float(distances[0][0])
        confidence = round((1 - distance) * 100, 2)
        print(f'   distance={distance:.4f} | confiance={confidence}%')

        return trajectory, distance, confidence