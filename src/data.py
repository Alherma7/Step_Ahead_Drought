"""Loading and saving the competition's flat CSV files."""
import pandas as pd

from src import config


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename release-specific column names to the canonical ones in config.

    Train.csv ships as sample_id/target, Test.csv as ID (no target column), and
    SampleSubmission.csv as ID/Target. Source: notebooks/00_official_starter_reference.ipynb
    (normalise_id_column), which documents this same inconsistency across releases.
    """
    rename = {}
    if "sample_id" in df.columns and config.ID_COL not in df.columns:
        rename["sample_id"] = config.ID_COL
    if "Target" in df.columns and config.TARGET_COL not in df.columns:
        rename["Target"] = config.TARGET_COL
    return df.rename(columns=rename) if rename else df


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Train.csv, Test.csv and SampleSubmission.csv from data/raw/."""
    train = _normalise_columns(pd.read_csv(config.TRAIN_PATH, parse_dates=[config.TIME_COL]))
    test = _normalise_columns(pd.read_csv(config.TEST_PATH, parse_dates=[config.TIME_COL]))
    sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    return train, test, sample_submission


def save_submission(ids: pd.Series, predictions, path) -> None:
    """Write a two-column ID/Target submission file in the required format."""
    submission = pd.DataFrame({config.ID_COL: ids, "Target": predictions})
    submission.to_csv(path, index=False)
