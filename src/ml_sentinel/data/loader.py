from pathlib import Path

import pandas as pd


def load_dataset(
    path: str | Path,
) -> pd.DataFrame:
    """Load a CSV dataset from disk."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)
