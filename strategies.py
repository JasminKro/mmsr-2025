import random
from enum import Enum
from abc import ABC, abstractmethod

class Modality(str, Enum):
    AUDIO = "audio"
    LYRICS = "lyrics"
    VIDEO = "video"
    AUDIO_AUDIO = "audio_audio"
    AUDIO_LYRICS = "audio_lyrics"
    AUDIO_VIDEO = "audio_video"
    LYRICS_LYRICS = "lyrics_lyrics"
    LYRICS_AUDIO = "lyrics_audio"
    LYRICS_VIDEO = "lyrics_video"
    VIDEO_AUDIO = "video_audio"
    VIDEO_LYRICS = "video_lyrics"
    VIDEO_VIDEO = "video_video"
    ALL = "audio_lyrics_video"

# helper directory
MODALITY_MAP = {
    Modality.AUDIO: ["audio"],
    Modality.LYRICS: ["lyrics"],
    Modality.VIDEO: ["video"],
    Modality.AUDIO_AUDIO: ["audio", "audio"],
    Modality.AUDIO_LYRICS: ["audio", "lyrics"],
    Modality.AUDIO_VIDEO: ["audio", "video"],
    Modality.LYRICS_AUDIO: ["lyrics", "audio"],
    Modality.LYRICS_LYRICS: ["lyrics", "lyrics"],
    Modality.LYRICS_VIDEO: ["lyrics", "video"],
    Modality.VIDEO_AUDIO: ["video", "audio"],
    Modality.VIDEO_LYRICS: ["lyrics", "video"],
    Modality.VIDEO_VIDEO: ["video", "video"],
    Modality.ALL: ["audio", "lyrics", "video"]
}

class RetrievalStrategy(ABC):
    @abstractmethod
    def search(self, query_id: str, k: int):
        #Returns (list_of_ids, list_of_scores)
        pass


class RandomStrategy(RetrievalStrategy):
    def __init__(self, rs_instance):
        self.rs = rs_instance

    def search(self, query_id, k):
        ids, metrics, scores = self.rs.retrieve(query_id=query_id, k_neighbors=k)
        print(ids, metrics, scores)
        return ids, metrics, scores


class UnimodalStrategy(RetrievalStrategy):
    def __init__(self, rs_instance, modality):
        self.rs = rs_instance
        self.modality = modality

    def search(self, query_id, k):
        self.rs.set_modality(self.modality)
        ids, metrics, scores = self.rs.retrieve(query_id=query_id, k_neighbors=k)
        return ids, metrics, scores


class EarlyFusionStrategy(RetrievalStrategy):
    def __init__(self, rs_instance, modalities):
        self.rs = rs_instance
        self.modalities = modalities

    def search(self, query_id, k):
        ids, metrics, scores = self.rs.retrieve(query_id=query_id, k_neighbors=k)
        return ids, metrics, scores


class LateFusionStrategy(RetrievalStrategy):
    def __init__(self, rs_instance, modalities):
        self.rs = rs_instance
        self.modalities = modalities

    def search(self, query_id, k):
        self.rs.set_modality(self.modalities)
        ids, metrics, scores = self.rs.retrieve(query_id=query_id, k_neighbors=k)
        return ids, metrics, scores


class NeuralNetworkStrategy(RetrievalStrategy):
    def __init__(self, rs_instance, modalities):
        self.rs = rs_instance
        self.modalities = modalities

    def search(self, query_id, k):
        ids, metrics, scores = self.rs.retrieve(query_id=query_id, k_neighbors=k)
        return ids, metrics, scores