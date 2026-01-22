import random
from enum import Enum
from abc import ABC, abstractmethod

class Modality(str, Enum):
    AUDIO = "audio"
    LYRICS = "lyrics"
    VIDEO = "video"
    AUDIO_LYRICS = "audio_lyrics"
    AUDIO_VIDEO = "audio_video"
    LYRICS_VIDEO = "lyrics_video"
    ALL = "all"

# helper directory
MODALITY_MAP = {
    Modality.AUDIO: ["audio"],
    Modality.LYRICS: ["lyrics"],
    Modality.VIDEO: ["video"],
    Modality.AUDIO_LYRICS: ["audio", "lyrics"],
    Modality.AUDIO_VIDEO: ["audio", "video"],
    Modality.LYRICS_VIDEO: ["lyrics", "video"],
    Modality.ALL: ["audio", "lyrics", "video"]
}


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
    def __init__(self, rs_instance, modality):
        self.rs = rs_instance
        self.modality = modality

    def search(self, query_id, k):
        self.rs.set_modality(self.modality)
        ids, metrics = self.rs.retrieve(query_id=query_id, k_neighbors=k)

        # Extract scores from the metrics dictionary
        # We use .get() as a fallback in case 'cosine_sim' isn't the key
        scores = metrics.get("cosine_sim", [0.0] * len(ids))
        return ids, scores

class EarlyFusionStrategy(RetrievalStrategy):
    def __init__(self, rs_instance, modalities):
        self.rs = rs_instance
        self.modalities = modalities

    def search(self, query_id, k):
        ids, metrics = self.rs.retrieve(query_id=query_id, k_neighbors=k)

        # Extract scores from the metrics dictionary
        # We use .get() as a fallback in case 'cosine_sim' isn't the key
        scores = metrics.get("cosine_sim", [0.0] * len(ids))
        return ids, scores

class LateFusionStrategy(RetrievalStrategy):
    def __init__(self, rs_instance, modalities):
        self.rs = rs_instance
        self.modalities = modalities

    def search(self, query_id, k):
        ids, metrics = self.rs.retrieve(query_id=query_id, k_neighbors=k)

        # Extract scores from the metrics dictionary
        # We use .get() as a fallback in case 'cosine_sim' isn't the key
        scores = metrics.get("cosine_sim", [0.0] * len(ids))
        return ids, scores