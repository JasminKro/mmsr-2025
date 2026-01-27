import numpy as np
from common import Evaluator, evaluate_system


class RandomBaselineRetrievalSystem:
    """
    Random baseline:
      - rankings(query_id) -> random scores for all tracks (length N)
    """

    def __init__(self, evaluator, seed=0):
        self.evaluator = evaluator
        self.rng = np.random.default_rng(seed=seed)

    def rankings(self, query_id):
        return self.rng.uniform(-1.0, 1.0, size=(len(self.evaluator.ids),))

    # optional convenience 
    def retrieve(self, query_id, k_neighbors):
        assert k_neighbors > 0
        rankings = self.rankings(query_id)  # random ranks
        metrics = {
            f"Precision@{k_neighbors}": self.evaluator.precision(query_id, rankings, k_neighbors),
            f"Recall@{k_neighbors}": self.evaluator.recall(query_id, rankings, k_neighbors),
            f"MRR@{k_neighbors}": self.evaluator.mrr(query_id, rankings, k_neighbors),
            f"nDCG@{k_neighbors}": self.evaluator.ndcg(query_id, rankings, k_neighbors),
        }

        rand_indices = self.rng.integers(0, len(self.evaluator.ids), k_neighbors)
        rand_ids = [self.evaluator.ids[idx] for idx in rand_indices]
        scores = [rankings[idx] for idx in rand_indices]

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        scores = [scores[i] for i in order]
        rand_ids = [rand_ids[i] for i in order]
        rand_indices = [rand_indices[i] for i in order]

        return rand_ids, metrics, scores


if __name__ == "__main__":
    data_root = "./data"
    evaluator = Evaluator(data_root)

    rs = RandomBaselineRetrievalSystem(evaluator, seed=0)

    ids, metrics, scores = rs.retrieve(query_id="NDroPROgWm3jBxjH", k_neighbors=5)
    print("ids:", ids)
    print("metrics:", metrics)
    print("scores:", scores)

    # print("\n=== RANDOM BASELINE EVALUATION ===")
    # evaluate_system(evaluator, rs, k=10)
