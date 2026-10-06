from pathlib import Path

import numpy as np

from gino.data.dataset import EvolutionDataset, load_processed_dataset


def test_processed_test_pairs_have_chainable_futures():
    dataset_path = Path(__file__).resolve().parents[1] / "processed_data" / "particle_evolution_dataset.npz"
    data = load_processed_dataset(dataset_path)
    pair_ids = np.asarray(data["test_pair_ids"], dtype=np.int64)[:3]
    assert len(pair_ids) == 3, "Need at least three test pairs; re-run scripts/preprocess.py."
    names = [str(name) for name in data["feature_names"]]
    input_features = [name for name in ("u_x", "u_y", "u_z") if name in names]
    ds = EvolutionDataset(data, pair_ids, input_features, [], rollout_horizon=4)
    for pair_id in pair_ids:
        length = ds.chain_length(int(pair_id))
        assert length >= 2, (
            f"Pair {pair_id} has no chainable future pairs. "
            "Re-run scripts/preprocess.py after identity-tracking fix."
        )
