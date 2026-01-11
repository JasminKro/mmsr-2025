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
    "genres": "id_genres_mmsr.tsv",
    "popularity": "id_total_listens.tsv" # needed to calcluate pop@k
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

        # total listens / popularity  (JETZT erst möglich, weil self.ids existiert)
        df_pop = pd.read_csv(os.path.join(data_root, DATAFRAMES["popularity"]), sep="\t")

        # robust: nimm die erste Spalte, die nicht "id" ist
        listen_col = [c for c in df_pop.columns if c != "id"][0]

        # align popularity to the same ID order as evaluator.ids
        df_pop = df_pop.set_index("id").reindex(self.ids).reset_index()
        self.listens = df_pop[listen_col].fillna(0).to_numpy(dtype=np.float64)

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

    # -------------------------
    # POP Metric
    # -------------------------
    def pop_at_k(self, query_id, rankings, k, log1p=True):
        """
        Pop@k: average popularity (total_listens) of the Top-k retrieved items.
        If log1p=True, uses log(1+listens) to reduce heavy-tail skew.
        """
        query_index = self.ids.index(query_id)

        # remove query itself
        rankings_ = np.delete(rankings, query_index)
        listens_ = np.delete(self.listens, query_index)

        top_idx = np.argsort(rankings_)[::-1][:k]
        vals = listens_[top_idx]

        if log1p:
            vals = np.log1p(vals)

        return float(np.mean(vals))

    
    # -------------------------
    # JACCARD-THRESHOLD METRICS
    # -------------------------
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
    
    # -------------------------
    # OVERLAP METRICS (as defined in slide 12)
    # relevant iff |Gq ∩ Gr| > 0  <=>  jaccard > 0
    # -------------------------
    def _binary_relevance_overlap(self, query_id, rankings, k):
        # overlap means at least one shared genre
        relevance = (self.jaccard[query_id] > 0.0)  # binary overlap

        query_index = self.ids.index(query_id)
        rankings = np.delete(rankings, query_index)
        relevance = np.delete(relevance, query_index)

        retrieved_indices = np.argsort(rankings)[::-1][:k]
        return relevance, retrieved_indices

    def precision_overlap(self, query_id, rankings, k):
        relevance, retrieved_indices = self._binary_relevance_overlap(query_id, rankings, k)
        retrieved_relevant = relevance[retrieved_indices].astype(np.float64).sum()
        return retrieved_relevant / len(retrieved_indices)

    def recall_overlap(self, query_id, rankings, k):
        relevance, retrieved_indices = self._binary_relevance_overlap(query_id, rankings, k)
        retrieved_relevant = relevance[retrieved_indices].astype(np.float64).sum()
        all_relevant = relevance.astype(np.float64).sum()
        if all_relevant > 0:
            return retrieved_relevant / all_relevant
        return None

    def mrr_overlap(self, query_id, rankings, k):
        relevance, retrieved_indices = self._binary_relevance_overlap(query_id, rankings, k)
        for i, idx in enumerate(retrieved_indices):
            if relevance[idx]:
                return 1.0 / (i + 1)
        return 0.0

    def ndcg_overlap(self, query_id, rankings, k):
        # binary relevance for nDCG based on overlap (0/1)
        relevance = (self.jaccard[query_id] > 0.0).astype(np.float64)

        query_index = self.ids.index(query_id)
        rankings = np.delete(rankings, query_index)
        relevance = np.delete(relevance, query_index)

        relevance = np.expand_dims(relevance, 0)
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

    # Jaccard-threshold metrics
    precisions, recalls, mrrs, ndcgs = [], [], [], []
    # Overlap metrics
    precisions_o, recalls_o, mrrs_o, ndcgs_o = [], [], [], []

    pops = []

    for id in tqdm(evaluator.ids, desc=f"Evaluating {rs.__class__.__name__}", unit="track"):

        rankings = rs.rankings(query_id=id)

        # Jaccard-threshold metrics 
        precision = evaluator.precision(query_id=id, rankings=rankings, k=k)
        mrr = evaluator.mrr(query_id=id, rankings=rankings, k=k)
        recall = evaluator.recall(query_id=id, rankings=rankings, k=k)
        ndcg = evaluator.ndcg(query_id=id, rankings=rankings, k=k)

        precisions.append(precision)
        mrrs.append(mrr)
        if recall is not None:
            recalls.append(recall)
        ndcgs.append(ndcg)

        # Overlap (one common genre is enough) metrics
        p2 = evaluator.precision_overlap(query_id=id, rankings=rankings, k=k)
        r2 = evaluator.recall_overlap(query_id=id, rankings=rankings, k=k)
        m2 = evaluator.mrr_overlap(query_id=id, rankings=rankings, k=k)
        n2 = evaluator.ndcg_overlap(query_id=id, rankings=rankings, k=k)

        precisions_o.append(p2)
        mrrs_o.append(m2)
        ndcgs_o.append(n2)
        if r2 is not None:
            recalls_o.append(r2)

        # popularity 
        pops.append(evaluator.pop_at_k(query_id=id, rankings=rankings, k=k, log1p=True))

    # Compute statistics
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    mrrs = np.array(mrrs)
    ndcgs = np.array(ndcgs)

    precisions_o = np.array(precisions_o)
    recalls_o = np.array(recalls_o)
    mrrs_o = np.array(mrrs_o)
    ndcgs_o = np.array(ndcgs_o)

    pops = np.array(pops, dtype=np.float64)


    # Print Jaccard-threshold results
    print("\nJACCARD-THRESHOLD RELEVANCE")
    print("METRIC       MEAN   |  STD")
    print("-----------------------------")
    print(f"Precision@{k}: {precisions.mean():.4f} | {precisions.std():.4f}")
    print(f"Recall@{k}:    {recalls.mean():.4f} | {recalls.std():.4f}")
    print(f"MRR@{k}:       {mrrs.mean():.4f} | {mrrs.std():.4f}")
    print(f"nDCG@{k}:      {ndcgs.mean():.4f} | {ndcgs.std():.4f}")

    # Print overlap results
    print("\nOVERLAP RELEVANCE (slide rule)")
    print("METRIC       MEAN   |  STD")
    print("-----------------------------")
    print(f"Precision@{k}: {precisions_o.mean():.4f} | {precisions_o.std():.4f}")
    print(f"Recall@{k}:    {recalls_o.mean():.4f} | {recalls_o.std():.4f}")
    print(f"MRR@{k}:       {mrrs_o.mean():.4f} | {mrrs_o.std():.4f}")
    print(f"nDCG@{k}:      {ndcgs_o.mean():.4f} | {ndcgs_o.std():.4f}")

    # Print popularity TODO: Later include coverage here
    print("\nPOPULARITY")
    print("METRIC            MEAN   |  STD")
    print("--------------------------------")
    print(f"Pop@{k} (log1p): {pops.mean():.4f} | {pops.std():.4f}")

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


'''
Outsourced this to jupyter notebook -> delete code later 

if __name__ == "__main__":

    from baseline import RandomBaselineRetrievalSystem
    from unimodal import UnimodalRetrievalSystem
    from early_fusion import EarlyFusionRetrievalSystem
    from late_fusion import LateFusionRetrievalSystem

    data_dir = "./data"
    k = 

    evaluator = Evaluator(data_dir, jaccard_relevant_threshold=0.25)

    def banner(title):
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)

    # 1) Random baseline
    banner("Random Baseline")
    rs = RandomBaselineRetrievalSystem(evaluator, seed=0)
    evaluate_system(evaluator, rs, k=k)

    # 2) Unimodal (each modality)
    unimodal_rs = UnimodalRetrievalSystem(data_dir, evaluator)
    for modality in MODALITIES:
        banner(f"Unimodal | modality={modality}")
        unimodal_rs.set_modality(modality)
        evaluate_system(evaluator, unimodal_rs, k=k)

    # 3) Early Fusion 
    rng = np.random.default_rng(0)
    two_mods = rng.choice(MODALITIES, size=2, replace=False).tolist()

    early_rs = EarlyFusionRetrievalSystem(data_dir, evaluator)

    # test it once with all three modalities and then with random two 
    for label, mods in [
        ("ALL modalities", ["audio", "lyrics", "video"]),
        (f"RANDOM 2 modalities = {two_mods}", two_mods),
    ]:
        banner(f"Early Fusion | {label} | modalities={mods}")
        early_rs.set_modalities(mods)
        evaluate_system(evaluator, early_rs, k=k)

    # 4) Late fusion (strategies x modality sets)
    rng = np.random.default_rng(0)
    two_mods = rng.choice(MODALITIES, size=2, replace=False).tolist()

    # test it once with all three modalities and then with random two 
    modality_sets = [
        ("ALL modalities", ["audio", "lyrics", "video"]),
        (f"RANDOM 2 modalities = {two_mods}", two_mods),
    ]

    late_configs = [
        ("RRF", dict(fusion="rrf", rrf_k=60)),

        ("norm_sum (zscore + equal)", dict(fusion="norm_sum", norm="zscore", weighting="equal")),
        ("norm_sum (zscore + auto_agreement)", dict(fusion="norm_sum", norm="zscore", weighting="auto")),

        ("norm_sum (minmax + equal)", dict(fusion="norm_sum", norm="minmax", weighting="equal")),
        ("norm_sum (minmax + auto_agreement)", dict(fusion="norm_sum", norm="minmax", weighting="auto")),
    ]

    for mod_label, mods in modality_sets:
        for cfg_label, cfg in late_configs:
            banner(f"Late Fusion | {cfg_label} | {mod_label} | modalities={mods}")
            rs = LateFusionRetrievalSystem(data_dir, evaluator, modalities=mods, **cfg)
            evaluate_system(evaluator, rs, k=k)

'''