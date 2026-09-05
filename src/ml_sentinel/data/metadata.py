from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata describing a dataset."""

    name: str
    version: str
    scenario: str
    created_at: datetime
    row_count: int
