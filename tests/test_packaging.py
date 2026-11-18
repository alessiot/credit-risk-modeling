import json

import pytest
from sklearn.linear_model import LogisticRegression

from credit_risk.packaging import file_sha256, load_artifact, save_artifact


def fitted() -> LogisticRegression:
    return LogisticRegression().fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])


def test_roundtrip(tmp_path):
    model = fitted()
    out = save_artifact(model, {"name": "m"}, tmp_path / "art")
    assert (out / "model.joblib").exists()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["name"] == "m"
    assert meta["model_sha256"] == file_sha256(out / "model.joblib")
    assert {"python", "numpy", "pandas", "scikit-learn", "lightgbm", "joblib"} <= set(
        meta["library_versions"]
    )
    assert "T" in meta["created_utc"]
    loaded, meta2 = load_artifact(out)
    assert meta2 == meta
    assert loaded.predict_proba([[1.5]])[0, 1] == model.predict_proba([[1.5]])[0, 1]


def test_refuses_overwrite_but_allows_empty_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    save_artifact(fitted(), {}, tmp_path / "empty")
    with pytest.raises(FileExistsError):
        save_artifact(fitted(), {}, tmp_path / "empty")


def test_detects_tampering(tmp_path):
    out = save_artifact(fitted(), {}, tmp_path / "art")
    with (out / "model.joblib").open("ab") as fh:
        fh.write(b"x")
    with pytest.raises(ValueError, match="sha256"):
        load_artifact(out)


@pytest.mark.parametrize("key", ["created_utc", "library_versions", "model_sha256"])
def test_reserved_metadata_keys(tmp_path, key):
    with pytest.raises(ValueError, match=key):
        save_artifact(fitted(), {key: "x"}, tmp_path / "art")
