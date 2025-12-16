import numpy as np
import pandas as pd


class RandomBaselineRetrievalSystem:
    """
    random baseline retrieval system.

    Regardless of the query track, this system randomly selects
    tracks from the entire dataset
    """

    def __init__(self, data_root):
        """
        Loads a global list of all track IDs.
        """
        # Any file that contains all track ids works
        df = pd.read_csv(f"{data_root}/id_information_mmsr.tsv", sep="\t")
        self.all_ids = df["id"].tolist()

    def retrieve(self, query_id, k_neighbors):
        """
        Returns:
            ids: list of randomly selected track IDs
            scores: random scores in [0, 1)
        """
        if query_id not in self.all_ids:
            raise ValueError(f"Query id {query_id} not found in dataset.")

        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be > 0")

        # Remove query track
        candidate_ids = [tid for tid in self.all_ids if tid != query_id]

        if k_neighbors > len(candidate_ids):
            raise ValueError(
                f"k_neighbors ({k_neighbors}) is larger than "
                f"available candidates ({len(candidate_ids)})"
            )

        # Random sample without replacement
        sampled_ids = np.random.choice(
            candidate_ids,
            size=k_neighbors,
            replace=False
        ).tolist()

        # Random scores (purely for interface compatibility)
        random_scores = np.random.rand(k_neighbors).tolist()

        return sampled_ids, random_scores


if __name__ == "__main__":
    data_root = "./data"
    query_id = "NDroPROgWm3jBxjH"

    rs = RandomBaselineRetrievalSystem(data_root)
    ids, scores = rs.retrieve(query_id=query_id, k_neighbors=5)

    print("Random baseline IDs:", ids)
    print("Random baseline scores:", scores)