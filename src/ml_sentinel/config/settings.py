from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"

REFERENCE_DATA_DIR = DATA_DIR / "reference"
PRODUCTION_DATA_DIR = DATA_DIR / "production"

REFERENCE_DATA_PATH = REFERENCE_DATA_DIR / "reference.csv"
PRODUCTION_DATA_PATH = PRODUCTION_DATA_DIR / "production.csv"
