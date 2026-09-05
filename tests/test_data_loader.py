import pandas as pd
import pytest

from ml_sentinel.data.loader import load_dataset


def test_load_dataset(tmp_path):
    path = tmp_path / "dataset.csv"

    original = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    original.to_csv(path, index=False)

    loaded = load_dataset(path)

    pd.testing.assert_frame_equal(
        original,
        loaded,
    )


def test_load_dataset_missing_file(tmp_path):
    path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_dataset(path)
