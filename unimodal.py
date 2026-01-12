import os
import pandas as pd
import numpy as np
from numpy.linalg import norm

from common import Evaluator, DATAFRAMES, MODALITIES, evaluate_system


class UnimodalRetrievalSystem:

    def __init__(self, data_root, evaluator):

        self.data_root = data_root
        self.evaluator = evaluator
        print("loading...")
        self.features, self.index_to_id, self.id_to_index = self._load_features(MODALITIES)
        self.modality = None

    def _load_dataframe(self, modality):
        filename, start_column = DATAFRAMES[modality]
        df = pd.read_csv(os.path.join(self.data_root, filename), sep="\t")
        return df, start_column

    def _load_features(self, modalities, normalize=True):
        # Load audio dataframe to use its ID order as the reference and
        df_audio, _ = self._load_dataframe("audio")
        ref_order = df_audio["id"]

        features = {}
        for modality in modalities:
            df, start_column = self._load_dataframe(modality)
            df = df.set_index("id").reindex(ref_order).reset_index()  # reorder current modality to exactly match the reference ID order 
            features_matrix = df.loc[:, start_column:].to_numpy()     # convert feature vectors to numpy array
            if normalize:
                features_matrix = features_matrix / norm(features_matrix, axis=1).reshape(-1, 1)  # normalized feature vectors (make each vector unit length)
            features[modality] = features_matrix

        index_to_id = df.id.to_list()  # index-to-id list
        id_to_index = dict(zip(df.id.to_list(), range(len(df))))  # id-to-index map
        return features, index_to_id, id_to_index

    def set_modality(self, modality):
        assert modality in MODALITIES, "invalid modality: " + modality
        self.modality = modality

    def rankings(self, query_id):
        assert self.modality is not None, "please set the modality with the `set_modality` method"
        features = self.features[self.modality]
        query_index = self.id_to_index[query_id]  # convert id to index
        query_vector = features[query_index]  # fetch query vector
        return features @ query_vector  # compute cosine similarities to all other vectors

    def retrieve(self, query_id, k_neighbors):
        assert k_neighbors > 0
        rankings = self.rankings(query_id)
        top_indices = np.argsort(rankings)[::-1][:k_neighbors+1].tolist()
        top_ids = [self.index_to_id[idx] for idx in top_indices]  # convert indices to ids
        top_ids.remove(query_id)  # remove query from retrieved items
        metrics = {
            f"Precision@{k_neighbors}": self.evaluator.precision(query_id, rankings, k_neighbors),
            f"Recall@{k_neighbors}": self.evaluator.recall(query_id, rankings, k_neighbors),
            f"MRR@{k_neighbors}": self.evaluator.mrr(query_id, rankings, k_neighbors),
            f"nDCG@{k_neighbors}": self.evaluator.ndcg(query_id, rankings, k_neighbors),
        }
        return top_ids, metrics


if __name__ == "__main__":

    data_root = "./data"
    evaluator = Evaluator(data_root)
    unimodal_rs = UnimodalRetrievalSystem(data_root, evaluator)
    #unimodal_rs.set_modality("audio")  # new method added

    #ids, metrics = unimodal_rs.retrieve(query_id="NDroPROgWm3jBxjH", k_neighbors=5)  # `modality` argument removed; returns metrics dictionary instead of cosine similarity list
    #print("ids:", ids)
    #print("metrics:", metrics)

    k = 10

    for modality in ["audio", "lyrics", "video"]:
        print("\n" + "=" * 60)
        print(f"Unimodal {modality}")
        unimodal_rs.set_modality(modality)
        evaluate_system(evaluator, unimodal_rs, k=k)



