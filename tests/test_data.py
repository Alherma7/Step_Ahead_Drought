import pandas as pd

from src.data import _normalise_columns, save_submission


def test_normalise_columns_renames_train_release_names():
    df = pd.DataFrame({"sample_id": ["a"], "time": ["2020-01-01"], "target": [1.0]})
    out = _normalise_columns(df)
    assert list(out.columns) == ["ID", "time", "target"]


def test_normalise_columns_renames_submission_target():
    df = pd.DataFrame({"ID": ["a"], "Target": [1.0]})
    out = _normalise_columns(df)
    assert list(out.columns) == ["ID", "target"]


def test_normalise_columns_is_noop_when_already_canonical():
    df = pd.DataFrame({"ID": ["a"], "time": ["2020-01-01"], "target": [1.0]})
    out = _normalise_columns(df)
    pd.testing.assert_frame_equal(out, df)


def test_save_submission_writes_id_and_target_columns(tmp_path):
    ids = pd.Series(["20150901_-55.5_-68.5", "20150901_-55.5_-67.5"])
    predictions = [0.1, -0.2]
    path = tmp_path / "submission.csv"

    save_submission(ids, predictions, path)

    written = pd.read_csv(path)
    assert list(written.columns) == ["ID", "Target"]
    assert len(written) == 2
