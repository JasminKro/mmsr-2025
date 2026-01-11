import os
import json
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA

from unimodal import UnimodalRetrievalSystem
from common import Evaluator, CACHE_DIR, evaluate_system


AUDIO_DIM = 768
LYRICS_DIM = 500
VIDEO_DIM = 4096*2
PROJ_DIM = 1024


class EarlyFusionRetrievalSystem(UnimodalRetrievalSystem):

    def __init__(self, data_root, evaluator):

        self.data_root = data_root
        self.evaluator = evaluator
        self.fused_features_path = os.path.join(CACHE_DIR, "fused_features.npz")
        self.id_mapping_path = os.path.join(CACHE_DIR, "id_mapping.json")

        if not (os.path.isfile(self.fused_features_path) and os.path.isfile(self.id_mapping_path)):
            print("cache not found | preprocessing features and building cache")
            self._build_cache()
        
        self._load_cache()

    def _build_cache(self):

        # Collect feature matrices of all modalities
        features, _, id_mapping = self._load_features(normalize=False)
        # Early fusion: concatenate all modalities along the feature dimension
        features_fused = np.concatenate(list(features.values()), axis=1)

        # Drop constant features to avoid division by zero (since std will be 0)
        std = features_fused.std(axis=0)
        features_fused = features_fused[:, std > 0]

        # Perform standardization (so each feature dimension has mean 0 and variance 1)
        mean = features_fused.mean(axis=0)
        std = features_fused.std(axis=0)
        features_fused = (features_fused - mean) / std 

        # Project fused features to a lower dimensional space using PCA
        # pca = PCA(n_components=PROJ_DIM, random_state=0)
        # features_fused = pca.fit_transform(features_fused)

        # Normalized feature vectors (make each vector unit length)
        features_fused = features_fused / norm(features_fused, axis=1).reshape(-1, 1)

        # Cache fused features to avoid recomputation in the future
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(self.fused_features_path, features=features_fused)

        # Cache the id mapping
        with open(self.id_mapping_path, "w") as f:
            json.dump(id_mapping, f)

    def _load_cache(self):
        self.features = np.load(self.fused_features_path)["features"]
        with open(self.id_mapping_path, "r") as f:
            self.id_to_index = json.load(f)

        # Compute reverse mapping
        self.index_to_id = [None] * len(self.id_to_index)
        for id, idx in self.id_to_index.items():
            self.index_to_id[idx] = id
    
    def rankings(self, query_id):
        features = self.features
        query_index = self.id_to_index[query_id]  # convert id to index
        query_vector = features[query_index]  # fetch query vector
        return features @ query_vector  # compute cosine similarities to all other vectors


if __name__ == "__main__":

    data_root = "./data"
    evaluator = Evaluator(data_root)
    early_fusion_rs = EarlyFusionRetrievalSystem(data_root, evaluator)

    #ids, metrics = early_fusion_rs.retrieve(query_id="NDroPROgWm3jBxjH", k_neighbors=5)  # returns metrics dictionary instead of cosine similarity list
    #print("ids:", ids)
    #print("metrics:", metrics)

    k = 10

    print("\n" + "=" * 60)
    print("Early Fusion")
    evaluate_system(evaluator, early_fusion_rs, k=k)
