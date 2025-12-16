import os
import json
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA

from unimodal import *


AUDIO_DIM = 768
LYRICS_DIM = 500
VIDEO_DIM = 4096*2
PROJ_DIM = 1024


class EarlyFusionRetrievalSystem(UnimodalRetrievalSystem):

    def __init__(self, data_root):

        self.data_root = data_root
        self.cache_dir = "./cache"
        self.fused_features_path = os.path.join(self.cache_dir, "fused_features.npz")
        self.id_mapping_path = os.path.join(self.cache_dir, "id_mapping.json")

        if not (os.path.isfile(self.fused_features_path) and os.path.isfile(self.id_mapping_path)):
            print("cache not found | preprocessing features and building cache")
            self._build_cache()
        
        self._load_cache()

    def _build_cache(self):

        # Collect feature matrices of all modalities
        all_features = []
        id_mapping = None

        for modality in MODALITIES:
            # Load raw (unnormalized) features for current modality 
            features, _, id_to_index = self._process_data(modality, normalize=False) 
            all_features.append(features)
            # Use audio as the reference ordering for all modalities
            if modality == "audio":
                id_mapping = id_to_index

        # Early fusion: concatenate all modalities along the feature dimension
        features_fused = np.concatenate(all_features, axis=1)

        # Drop constant features to avoid division by zero (since std will be 0)
        std = features_fused.std(axis=0)
        features_fused = features_fused[:, std > 0]

        # Perform standardization (so each feature dimension has mean 0 and variance 1)
        mean = features_fused.mean(axis=0)
        std = features_fused.std(axis=0)
        features_fused = (features_fused - mean) / std 

        # Project fused features to a lower dimensional space using PCA
        pca = PCA(n_components=PROJ_DIM, random_state=0)
        features_fused = pca.fit_transform(features_fused)

        # Normalized feature vectors (make each vector unit length)
        features_fused = features_fused / norm(features_fused, axis=1).reshape(-1, 1)

        # Cache fused features to avoid recomputation in the future
        os.makedirs(self.cache_dir, exist_ok=True)
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

    def retrieve(self, query_id, k_neighbors):
        return self._retrieve(query_id, k_neighbors, self.features, self.id_to_index, self.index_to_id)


if __name__ == "__main__":
    
    early_fusion_rs = EarlyFusionRetrievalSystem("./data")
    ids, scores = early_fusion_rs.retrieve(query_id="NDroPROgWm3jBxjH", k_neighbors=5)
    print("ids:", ids)
    print("scores:", scores)
