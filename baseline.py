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
        scores = self.rankings(query_id)
        top_idx = np.argsort(scores)[::-1][:k_neighbors + 1]
        top_ids = [self.evaluator.ids[i] for i in top_idx if self.evaluator.ids[i] != query_id][:k_neighbors]
        top_scores = [float(scores[self.evaluator.id_to_idx[tid]]) if hasattr(self.evaluator, "id_to_idx") else float(scores[i])
                      for i, tid in zip(top_idx, [self.evaluator.ids[i] for i in top_idx]) if tid != query_id][:k_neighbors]
        return top_ids, top_scores


if __name__ == "__main__":
    data_root = "./data"
    evaluator = Evaluator(data_root)

    rs = RandomBaselineRetrievalSystem(evaluator, seed=0)

    print("\n=== RANDOM BASELINE EVALUATION ===")
    evaluate_system(evaluator, rs, k=10)
