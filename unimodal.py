import os
import pandas as pd
import numpy as np
from numpy.linalg import norm


AUDIO_DIM = 768
LYRICS_DIM = 500
VIDEO_DIM = 4096*2


class UnimodalRetrievalSystem:

    def __init__(self, data_root):

        self.data_root = data_root
        self.modalities = ["audio", "lyrics", "video"]
        self.data = {modality:self._process_data(modality) for modality in self.modalities}

    def _process_data(self, modality):

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
        
        print(f"processing {modality} data...")

        df = pd.read_csv(os.path.join(self.data_root, filename), sep="\t")
        features_matrix = df.loc[:, start_column:].to_numpy()  # feature vectors as numpy array
        features_matrix_norm = features_matrix / norm(features_matrix, axis=1).reshape(-1, 1)  # normalized feature vectors (make each vector unit length)
        index_to_id = df.id.to_list()  # index-to-id list
        id_to_index = dict(zip(df.id.to_list(), range(len(df))))  # id-to-index map

        return features_matrix_norm, index_to_id, id_to_index

    def retrieve(self, query_id, modality, k_neighbors):

        assert modality in self.modalities, "invalid modality: " + modality
        assert k_neighbors > 0

        matrix, index_to_id, id_to_index = self.data[modality]  # load data
        query_index = id_to_index[query_id]  # convert id to index
        query_vector = matrix[query_index]  # fetch query vector
        similarity_scores = matrix @ query_vector  # compute cosine similarities to all other vectors
        top_indices = np.argsort(similarity_scores)[-k_neighbors-1:-1]  # sort similarities and select k_neighbors (skip the query itself)
        top_ids = [index_to_id[idx] for idx in top_indices]  # convert indices to ids

        return list(reversed(top_ids)), list(reversed(similarity_scores[top_indices].tolist()))



if __name__ == "__main__":
    
    # unimodal_rs = UnimodalRetrievalSystem("./data")
    # print()

    # query_id = "NDroPROgWm3jBxjH"
    # print("query:", query_id)

    # for modality in unimodal_rs.modalities:
    #     print("-"*10 + modality + "-"*10)
    #     ids, scores = unimodal_rs.retrieve(query_id, modality, 5)
    #     print("ids:", ids)
    #     print("scores", scores)


    unimodal_rs = UnimodalRetrievalSystem("./data")
    ids, scores = unimodal_rs.retrieve(query_id="NDroPROgWm3jBxjH", modality="audio", k_neighbors=5)
    print("ids:", ids)
    print("scores:", scores)

