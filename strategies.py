import random
from abc import ABC, abstractmethod

class RetrievalStrategy(ABC):
    @abstractmethod
    def search(self, query_id: str, k: int):
        #Returns (list_of_ids, list_of_scores)
        pass


class RandomStrategy(RetrievalStrategy):
    def __init__(self, all_ids):
        self.all_ids = all_ids

    def search(self, query_id, k):
        # Pick k random IDs from the dataset
        ids = random.sample(self.all_ids, min(k, len(self.all_ids)))
        scores = [0.0] * len(ids)
        return ids, scores


class UnimodalStrategy(RetrievalStrategy):
    def __init__(self, rs_instance):
        self.rs = rs_instance

    def search(self, query_id, k):
        # set modality first
        self.rs.set_modality("audio")
        ids, metrics = self.rs.retrieve(query_id=query_id, k_neighbors=k)

        # Extract scores from the metrics dictionary
        # We use .get() as a fallback in case 'cosine_sim' isn't the key
        scores = metrics.get("cosine_sim", [0.0] * len(ids))
        return ids, scores

