import os
import pandas as pd
import numpy as np
from numpy.linalg import norm


MODALITIES = ["audio", "lyrics", "video"]


class UnimodalRetrievalSystem:

    def __init__(self, data_root):

        self.data_root = data_root
        self.data = {modality:self._process_data(modality) for modality in MODALITIES}

    def _load_dataframe(self, modality):
        if modality == "audio":
            filename = "id_lyrics_bert_mmsr.tsv"
            start_column = "0"
        elif modality == "lyrics":
            filename = "id_mfcc_bow_mmsr.tsv"
            start_column = "mfccB000"
        elif modality == "video":
            filename = "id_vgg19_mmsr.tsv"
            start_column = "max0000"
        else:
            raise ValueError("invalid modality:", modality)
    
        df = pd.read_csv(os.path.join(self.data_root, filename), sep="\t")
        return df, start_column

    def _process_data(self, modality, normalize=True):
        df, start_column = self._load_dataframe(modality)

        if modality != "audio":
            # Load audio dataframe to use its ID order as the reference and
            # reorder current modality to exactly match the reference ID order 
            df_audio, _ = self._load_dataframe("audio")
            ref_order = df_audio["id"]
            df = df.set_index("id").reindex(ref_order).reset_index()

        features_matrix = df.loc[:, start_column:].to_numpy()  # feature vectors as numpy array
        if normalize:
            features_matrix = features_matrix / norm(features_matrix, axis=1).reshape(-1, 1)  # normalized feature vectors (make each vector unit length)
        index_to_id = df.id.to_list()  # index-to-id list
        id_to_index = dict(zip(df.id.to_list(), range(len(df))))  # id-to-index map

        return features_matrix, index_to_id, id_to_index
    
    def _retrieve(self, query_id, k_neighbors, features, id_to_index, index_to_id):
        assert k_neighbors > 0

        query_index = id_to_index[query_id]  # convert id to index
        query_vector = features[query_index]  # fetch query vector
        similarity_scores = features @ query_vector  # compute cosine similarities to all other vectors
        top_indices = np.argsort(similarity_scores)[-k_neighbors-1:-1]  # sort similarities and select k_neighbors (skip the query itself)
        top_ids = [index_to_id[idx] for idx in top_indices]  # convert indices to ids

        return list(reversed(top_ids)), list(reversed(similarity_scores[top_indices].tolist()))

    def retrieve(self, query_id, modality, k_neighbors):
        assert modality in MODALITIES, "invalid modality: " + modality
        features, index_to_id, id_to_index = self.data[modality]
        return self._retrieve(query_id, k_neighbors, features, id_to_index, index_to_id)


if __name__ == "__main__":
    
    # unimodal_rs = UnimodalRetrievalSystem("./data")
    # print()

    # query_id = "NDroPROgWm3jBxjH"
    # print("query:", query_id)

    # for modality in MODALITIES:
    #     print("-"*10 + modality + "-"*10)
    #     ids, scores = unimodal_rs.retrieve(query_id, modality, 5)
    #     print("ids:", ids)
    #     print("scores", scores)


    unimodal_rs = UnimodalRetrievalSystem("./data")
    ids, scores = unimodal_rs.retrieve(query_id="NDroPROgWm3jBxjH", modality="audio", k_neighbors=5)
    print("ids:", ids)
    print("scores:", scores)
