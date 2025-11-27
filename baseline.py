import numpy as np
from unimodal import UnimodalRetrievalSystem


# based on UnimodalRetrievalSystem 
class RandomBaselineRetrievalSystem(UnimodalRetrievalSystem): 
    """
    Random baseline: regardless of the query track, this system randomly
    selects tracks from the rest of the dataset (same modality).
    """

    def retrieve(self, query_id, modality, k_neighbors):
        """
        Returns:
            ids: list of randomly selected track IDs (length k_neighbors)
            scores: list of random scores in [0, 1) (same length), just to
                    keep the (ids, scores) structure compatible.
        """
        assert modality in self.modalities, "invalid modality: " + modality
        assert k_neighbors > 0

        # We dont need the feature matrix for random baseline (since we dont compute any similarities or similar), only the ID maps
        _, index_to_id, id_to_index = self.data[modality]

        n_items = len(index_to_id)

        # Index of the query track
        query_index = id_to_index[query_id]

        # All indices except the query itself
        all_indices = np.arange(n_items) # all possible indices
        candidate_indices = np.delete(all_indices, query_index)

        if k_neighbors > len(candidate_indices):
            raise ValueError(
                f"k_neighbors ({k_neighbors}) is larger than the number "
                f"of available items minus one ({len(candidate_indices)})"
            )

        # Randomly sample without replacement.
        # No fixed random seed -> new, different results each call.
        sampled_indices = np.random.choice(
            candidate_indices,
            size=k_neighbors,
            replace=False
        )

        # Convert indices to IDs
        sampled_ids = [index_to_id[i] for i in sampled_indices]

        random_scores = np.random.rand(k_neighbors).tolist()

        return sampled_ids, random_scores


if __name__ == "__main__":
    # Example use
    data_root = "./data"
    query_id = "NDroPROgWm3jBxjH"

    rs = RandomBaselineRetrievalSystem(data_root)
    ids, scores = rs.retrieve(query_id=query_id, modality="audio", k_neighbors=5)

    print("random baseline ids:   ", ids)
    print("random baseline scores:", scores)
