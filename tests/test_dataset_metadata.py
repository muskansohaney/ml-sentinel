from datetime import datetime, timezone

from ml_sentinel.data.metadata import DatasetMetadata


def test_dataset_metadata():
    created_at = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    metadata = DatasetMetadata(
        name="reference",
        version="1.0",
        scenario="normal",
        created_at=created_at,
        row_count=5000,
    )

    assert metadata.name == "reference"
    assert metadata.version == "1.0"
    assert metadata.scenario == "normal"
    assert metadata.row_count == 5000
