import os
import pandas as pd
import numpy as np
from numpy.linalg import norm

from common import Evaluator, evaluate_system


MODALITY_DESC = {
    "audio": "mfcc_bow",
    "video": "vgg19",
    "lyrics": "lyrics_bert",
}


class NeuralNetworkBasedRetrievalSystem:

    def __init__(self, data_root, evaluator, query_modality, result_modality):

        assert query_modality in MODALITY_DESC, "invalid query modality: " + query_modality
        assert result_modality in MODALITY_DESC, "invalid result modality: " + result_modality

        self.data_root = data_root
        self.evaluator = evaluator

        print("loading...")
        ref_df_path = os.path.join(self.data_root, "lyrics_bert_lyrics_bert_padding", "f1_lyrics_bert_f2_lyrics_bert_padding", "f1_lyrics_bert.tsv")
        ref_order = pd.read_csv(ref_df_path, sep="\t")["id"]
        self.index_to_id = ref_order.to_list() # index-to-id list
        self.id_to_index = dict(zip(self.index_to_id, range(len(self.index_to_id))))  # id-to-index map

        f1, f2 = self._load_data(query_modality, result_modality, ref_order)
        f1, f2 = self._preprocess_data(f1, f2)
        self.f1 = f1
        self.f2 = f2

    def _load_data(self, query_modality, result_modality, ref_order):

        desc1 = MODALITY_DESC[query_modality]
        desc2 = MODALITY_DESC[result_modality]

        # Assume primary order
        used1, used2 = desc1, desc2
        folder_path = os.path.join(self.data_root, f"{used1}_{used2}_padding", f"f1_{used1}_f2_{used2}_padding")
        filename1, filename2 = f"f1_{used1}.tsv", f"f2_{used1}.tsv"

        # Swap both folder and the descriptors used for filenames
        if not os.path.exists(folder_path):
            used1, used2 = desc2, desc1
            folder_path = os.path.join(self.data_root, f"{used1}_{used2}_padding", f"f1_{used1}_f2_{used2}_padding")
            filename1, filename2 = f"f2_{used1}.tsv", f"f1_{used1}.tsv"

        # Load feature vectors
        f1 = pd.read_csv(os.path.join(folder_path, filename1), sep="\t").set_index("id").reindex(ref_order).to_numpy()
        f2 = pd.read_csv(os.path.join(folder_path, filename2), sep="\t").set_index("id").reindex(ref_order).to_numpy()

        return f1, f2

    def _preprocess_data(self, f1, f2):

        # Normalize vectors (avoid division-by-zero)
        n1 = norm(f1, axis=1).reshape(-1, 1)
        n1[n1 == 0] = 1.0
        f1 = f1 / n1

        n2 = norm(f2, axis=1).reshape(-1, 1)
        n2[n2 == 0] = 1.0
        f2 = f2 / n2

        return f1, f2

    def rankings(self, query_id):
        query_features = self.f1
        result_features = self.f2
        query_index = self.id_to_index[query_id]  # convert id to index
        query_vector = query_features[query_index]  # fetch query vector
        return result_features @ query_vector  # compute cosine similarities to all other vectors

    def retrieve(self, query_id, k_neighbors):
        assert k_neighbors > 0
        rankings = self.rankings(query_id)
        top_indices = np.argsort(rankings)[::-1][:k_neighbors+1].tolist()
        top_ids = [self.index_to_id[idx] for idx in top_indices]  # convert indices to ids
        if query_id in top_ids:
            top_ids.remove(query_id)  # remove query from retrieved items
        top_ids = top_ids[:k_neighbors]  # ensure exactly k neighbors
        metrics = {
            f"Precision@{k_neighbors}": self.evaluator.precision(query_id, rankings, k_neighbors),
            f"Recall@{k_neighbors}": self.evaluator.recall(query_id, rankings, k_neighbors),
            f"MRR@{k_neighbors}": self.evaluator.mrr(query_id, rankings, k_neighbors),
            f"nDCG@{k_neighbors}": self.evaluator.ndcg(query_id, rankings, k_neighbors),
        }
        scores = [rankings[idx] for idx in top_indices[1:]]
        return top_ids, metrics, scores
    


class NeuralNetworkBasedRetrievalSystemExperiments(NeuralNetworkBasedRetrievalSystem):

    def __init__(self, data_root, evaluator, experiment_num):

        assert 1 <= experiment_num <= 6

        self.data_root = data_root
        self.evaluator = evaluator

        print("loading...")
        f1, f2 = self._load_data(experiment_num)
        f1, f2 = self._preprocess_data(f1, f2)
        self.f1 = f1
        self.f2 = f2

    
    def _load_data(self, experiment_num):

        folder_path = os.path.join(self.data_root, str(experiment_num))
        filename1, filename2 = "f1_mfcc_bow.tsv", "f2_mfcc_bow.tsv"

        # Load feature vectors
        df1 = pd.read_csv(os.path.join(folder_path, filename1), sep="\t")
        df2 = pd.read_csv(os.path.join(folder_path, filename2), sep="\t")
        f1 = df1.set_index("id").to_numpy()
        f2 = df2.set_index("id").to_numpy()

        self.index_to_id = df1["id"].to_list() # index-to-id list
        self.id_to_index = dict(zip(self.index_to_id, range(len(self.index_to_id))))  # id-to-index map

        return f1, f2



if __name__ == "__main__":

    evaluator = Evaluator("./data")
    # nn_rs = NeuralNetworkBasedRetrievalSystem(
    #     data_root="./data_nn/NN_pretrained_models_and_features/",
    #     evaluator=evaluator,
    #     query_modality="audio",
    #     result_modality="audio"
    # )
    # nn_rs = NeuralNetworkBasedRetrievalSystemExperimental(
    #     data_root="./data_nn/NN_experiments/",
    #     evaluator=evaluator,
    #     experiment_num=1
    # )

    # ids, metrics, scores = nn_rs.retrieve(query_id="NDroPROgWm3jBxjH", k_neighbors=5)  # `modality` argument removed; returns metrics dictionary instead of cosine similarity list
    # print("ids:", ids)
    # print("metrics:", metrics)
    # print("scores:", scores)
    
    k = 10
    for query_modality in ["audio", "lyrics", "video"]:
        for result_modality in ["audio", "lyrics", "video"]:
            nn_rs = NeuralNetworkBasedRetrievalSystem(
                data_root="./data_nn/NN_pretrained_models_and_features/",
                evaluator=evaluator,
                query_modality=query_modality,
                result_modality=result_modality
            )
            print("\n" + "=" * 60)
            print(f"Neural Network {query_modality}-{result_modality}")
            evaluate_system(evaluator, nn_rs, k=k)

    for experiment_num in range(1, 7):
        nn_rs = NeuralNetworkBasedRetrievalSystemExperiments(
            data_root="./data_nn/NN_experiments/",
            evaluator=evaluator,
            experiment_num=experiment_num
        )
        print("\n" + "=" * 60)
        print(f"Neural Network Experiment <{experiment_num}>")
        evaluate_system(evaluator, nn_rs, k=k)
