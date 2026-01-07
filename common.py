import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import ndcg_score

CACHE_DIR = "./cache"
MODALITIES = ["audio", "lyrics", "video"]
DATAFRAMES = {
    "audio": ("id_mfcc_bow_mmsr.tsv", "mfccB000"),
    "lyrics": ("id_lyrics_bert_mmsr.tsv", "0"),
    "video": ("id_vgg19_mmsr.tsv", "max0000"),
    "genres": "id_genres_mmsr.tsv"
}


class Evaluator:

    def __init__(self, data_root, jaccard_relevant_threshold=0.25):

        # Load the genre dataframe and use the audio dataframe's ID order as the reference order
        df_audio = pd.read_csv(os.path.join(data_root, DATAFRAMES["audio"][0]), sep="\t")
        ref_order = df_audio["id"]
        df = pd.read_csv(os.path.join(data_root, DATAFRAMES["genres"]), sep="\t")
        df = df.set_index("id").reindex(ref_order).reset_index()

        # Parse genre information
        ids = df.id.to_list()
        genres = [set(ast.literal_eval(genres_str)) for genres_str in df.genre.to_list()]
        self.genres = dict(zip(ids, genres))
        self.ids = ids

        # Compute jaccard scores for relevance
        self.jaccard_path = os.path.join(CACHE_DIR, "./jaccard.npz")
        if not os.path.isfile(self.jaccard_path):
            self._compute_jaccard()
        self.jaccard = np.load(self.jaccard_path)
        self.threshold = jaccard_relevant_threshold

    @staticmethod
    def _jaccard_index(query_genres, retrieved_genres):
        genres_inters = query_genres.intersection(retrieved_genres)
        genres_union = query_genres.union(retrieved_genres)
        return len(genres_inters) / len(genres_union)

    def _compute_jaccard(self):
        jaccard = {}
        for query_id in tqdm(self.ids, desc="Computing Jaccard Relevance"):  # loop over every possible query id
            values = [self._jaccard_index(self.genres[query_id], self.genres[ret_id]) for ret_id in self.ids]  # compute jaccard-index for every possible retrieved id
            jaccard[query_id] = np.array(values)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(self.jaccard_path, **jaccard)
    
    def _binary_relevance(self, query_id, rankings, k):
        # Compute binary relevance
        relevance = self.jaccard[query_id] >= self.threshold  # binary relevance

        # Remove query from rankings and relevances
        query_index = self.ids.index(query_id)
        rankings = np.delete(rankings, query_index)
        relevance = np.delete(relevance, query_index)

        retrieved_indices = np.argsort(rankings)[::-1][:k]
        return relevance, retrieved_indices

    def precision(self, query_id, rankings, k):
        relevance, retrieved_indices = self._binary_relevance(query_id, rankings, k)
        retrieved_relevant = relevance[retrieved_indices].astype(np.float64).sum()  # number of retrieved tracks which are relevant
        return retrieved_relevant / len(retrieved_indices)

    def recall(self, query_id, rankings, k):
        relevance, retrieved_indices = self._binary_relevance(query_id, rankings, k)
        retrieved_relevant = relevance[retrieved_indices].astype(np.float64).sum()
        all_relevant = relevance.astype(np.float64).sum()
        if all_relevant > 0:
            return retrieved_relevant / all_relevant
        return None

    def mrr(self, query_id, rankings, k):
        relevance, retrieved_indices = self._binary_relevance(query_id, rankings, k)
        for i, idx in enumerate(retrieved_indices):
            if relevance[idx]:
                return 1.0 / (i + 1)
        return 0.0
    
    def ndcg(self, query_id, rankings, k):
        relevance = self.jaccard[query_id]  # continuous/graded relevance

        # Remove query from rankings and relevances
        query_index = self.ids.index(query_id)
        rankings = np.delete(rankings, query_index)
        relevance = np.delete(relevance, query_index)
        
        # Compute nDCG score
        relevance = np.expand_dims(relevance, 0).astype(np.float64)
        rankings = np.expand_dims(rankings, 0)
        return ndcg_score(relevance, rankings, k=k)


class RandomRetrievalSystem:

    def __init__(self):
        self.rng = np.random.default_rng(seed=0)

    def rankings(self, query_id):
        return self.rng.uniform(-1, 1, (4148,))
    

def plot_hist(values, title, xlabel, bins=20):
    values = np.asarray(values)
    plt.figure()
    plt.hist(values, bins=bins)
    plt.axvline(values.mean(), linestyle="--", label=f"mean={values.mean():.3f}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_system(evaluator, rs, k):

    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []

    for id in tqdm(evaluator.ids, desc=f"Evaluating {rs.__class__.__name__}", unit="track"):

        rankings = rs.rankings(query_id=id)

        precision = evaluator.precision(query_id=id, rankings=rankings, k=k)
        mrr = evaluator.mrr(query_id=id, rankings=rankings, k=k)
        recall = evaluator.recall(query_id=id, rankings=rankings, k=k)
        ndcg = evaluator.ndcg(query_id=id, rankings=rankings, k=k)

        precisions.append(precision)
        mrrs.append(mrr)
        if recall is not None:
            recalls.append(recall)
        ndcgs.append(ndcg)

    # Compute statistics
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    mrrs = np.array(mrrs)
    ndcgs = np.array(ndcgs)

    print("METRIC       MEAN   |  STD")
    print("-----------------------------")
    print(f"Precision@{k}: {precisions.mean():.4f} | {precisions.std():.4f}")
    print(f"Recall@{k}:    {recalls.mean():.4f} | {recalls.std():.4f}")
    print(f"MRR@{k}:       {mrrs.mean():.4f} | {mrrs.std():.4f}")
    print(f"nDCG@{k}:      {ndcgs.mean():.4f} | {ndcgs.std():.4f}")

    # # Histograms
    # plot_hist(
    #     precisions,
    #     title=f"Precision@{k} distribution",
    #     xlabel=f"Precision@{k}",
    # )

    # if len(recalls) > 0:
    #     plot_hist(
    #         recalls,
    #         title=f"Recall@{k} distribution",
    #         xlabel=f"Recall@{k}",
    #     )

    # plot_hist(
    #     mrrs,
    #     title=f"MRR@{k} distribution",
    #     xlabel=f"MRR@{k}",
    # )

    # plot_hist(
    #     ndcgs,
    #     title=f"nDCG@{k} distribution",
    #     xlabel=f"nDCG@{k}",
    # )



if __name__ == "__main__":

    from unimodal import UnimodalRetrievalSystem
    from early_fusion import EarlyFusionRetrievalSystem

    data_dir = "./data"

    evaluator = Evaluator(data_dir)

    # Random baseline
    # rs = RandomRetrievalSystem()

    # Unimodal
    # rs = UnimodalRetrievalSystem(data_dir, evaluator)
    # rs.set_modality("audio")

    # Early Fusion
    rs = EarlyFusionRetrievalSystem(data_dir, evaluator)

    # Evaluate on all possible queries
    evaluate_system(evaluator, rs, k=5)
