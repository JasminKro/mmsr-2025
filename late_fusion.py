import numpy as np

from unimodal import UnimodalRetrievalSystem
from common import MODALITIES, Evaluator, evaluate_system


class LateFusionRetrievalSystem(UnimodalRetrievalSystem):
    """
    Late Fusion Retrieval System:
      - rankings(query_id) -> np.ndarray of length N (same order as evaluator.ids)
      - retrieve(query_id, k_neighbors) -> (top_ids, metrics dict)

    Fusion modes:
      - fusion="rrf": Reciprocal Rank Fusion (rank-based)
      - fusion="norm_sum": normalize scores per modality and sum (weighted)

    Weighting (only for norm_sum):
      - weighting="equal": equal weights
      - weighting="auto": agreement-based weights (Jaccard overlap of TopL sets)
    """

    def __init__(
        self,
        data_root,
        evaluator,
        modalities=None,
        fusion="rrf",
        norm="zscore",
        weighting="equal",
        rrf_k=60, # typical value for k
        topL=200,
        alpha=2.0,
        eps=1e-12,
    ):
        super().__init__(data_root, evaluator)

        self.used_modalities = list(modalities) if modalities is not None else list(MODALITIES)

        for m in self.used_modalities:
            assert m in MODALITIES, f"invalid modality: {m}"

        self.fusion = fusion
        self.norm = norm
        self.weighting = weighting
        self.rrf_k = int(rrf_k)
        self.topL = int(topL)
        self.alpha = float(alpha)
        self.eps = float(eps)

    # normalization helper methods
    @staticmethod
    def _zscore(x, eps=1e-12):
        x = np.asarray(x, dtype=float)
        return (x - x.mean()) / (x.std() + eps)

    @staticmethod
    def _minmax(x, eps=1e-12):
        x = np.asarray(x, dtype=float)
        mn, mx = float(x.min()), float(x.max())
        return (x - mn) / (mx - mn + eps)

    # agreement based weighting 
    def _auto_weights_by_agreement(self, scores_by_modality):
        """
        Compute weights w_m from agreement between modalities:
          1) take TopL indices per modality
          2) confidence_m = average Jaccard overlap with other TopL sets
          3) weights = normalize(confidence^alpha)

        alpha > 1 => sharper (more decisive) weighting.
        """
        M = len(scores_by_modality)
        if M < 2:
            return np.array([1.0], dtype=float)

        # TopL sets
        top_sets = []
        for s in scores_by_modality:
            s = np.asarray(s, dtype=float)
            L = min(self.topL, s.size)
            # indices of TopL (unordered)
            top_idx = np.argpartition(-s, kth=L - 1)[:L]
            top_sets.append(set(top_idx.tolist()))

        conf = np.zeros(M, dtype=float)
        for m in range(M):
            overlaps = []
            for j in range(M):
                if j == m:
                    continue
                A, B = top_sets[m], top_sets[j]
                inter = len(A & B)
                union = len(A | B) + self.eps
                overlaps.append(inter / union)
            conf[m] = float(np.mean(overlaps)) if overlaps else 0.0

        # fallback: no agreement -> equal weights
        if np.all(conf <= self.eps):
            w = np.ones(M, dtype=float)
            return w / w.sum()

        conf = np.maximum(conf, 0.0) ** self.alpha
        return conf / (conf.sum() + self.eps)

    # ranking for common.evaluate_system
    def rankings(self, query_id):
        """
        Returns fused scores for ALL tracks (length N) in evaluator.ids order.
        """
        assert query_id in self.id_to_index, "query_id not in index"

        N = len(self.index_to_id)
        q_idx = self.id_to_index[query_id]

        # per-modality score arrays
        scores_by_modality = []
        for m in self.used_modalities:
            X = self.features[m]             # shape (N, D)
            q = X[q_idx]                     # shape (D,)
            s = X @ q                        # shape (N,)
            scores_by_modality.append(s)

        # fusion 
        if self.fusion == "rrf":
            fused = np.zeros(N, dtype=float)
            for s in scores_by_modality:
                order = np.argsort(-s)  # descending
                ranks = np.empty_like(order)
                ranks[order] = np.arange(1, N + 1)  # 1..N
                fused += 1.0 / (self.rrf_k + ranks.astype(float))
            return fused

        if self.fusion == "norm_sum":
            # normalize each modality scores to comparable scale
            if self.norm == "zscore":
                normed = [self._zscore(s, self.eps) for s in scores_by_modality]
            elif self.norm == "minmax":
                normed = [self._minmax(s, self.eps) for s in scores_by_modality]
            else:
                raise ValueError("norm must be 'zscore' or 'minmax'")

            # choose weights
            if self.weighting == "equal":
                w = np.ones(len(normed), dtype=float)
                w /= w.sum()
            elif self.weighting == "auto":
                w = self._auto_weights_by_agreement(scores_by_modality)
            else:
                raise ValueError("weighting must be 'equal' or 'auto'")

            fused = np.zeros(N, dtype=float)
            for wi, si in zip(w, normed):
                fused += wi * si
            return fused

        raise ValueError("fusion must be 'rrf' or 'norm_sum'")

    # retrieve + metrics (like unimodal/early_fusion) for one query 
    def retrieve(self, query_id, k_neighbors):
        rankings = self.rankings(query_id=query_id)
        top_indices = np.argsort(rankings)[::-1][:k_neighbors + 1].tolist()
        top_ids = [self.index_to_id[idx] for idx in top_indices]
        top_ids.remove(query_id)

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

    k = 10

    experiments = [
        ("LateFusion RRF", dict(fusion="rrf", rrf_k=60)),
        ("LateFusion norm_sum (zscore + equal)", dict(fusion="norm_sum", norm="zscore", weighting="equal")),
        ("LateFusion norm_sum (zscore + auto_agree)", dict(fusion="norm_sum", norm="zscore", weighting="auto")),
    ]

    for name, cfg in experiments:
        print("\n" + "=" * 60)
        print(name)
        rs = LateFusionRetrievalSystem(data_root, evaluator, modalities=["audio", "lyrics", "video"], **cfg)
        evaluate_system(evaluator, rs, k=k)
