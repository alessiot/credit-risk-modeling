"""Save and load a model artifact with metadata."""

import hashlib
import json
import logging
import platform
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

_MODEL_FILE = "model.joblib"
_META_FILE = "metadata.json"
_LIBRARIES = ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib")


def file_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_artifact(model: object, metadata: Mapping[str, Any], out_dir: Path) -> Path:
    """Write the model and its metadata to a new directory.

    Args:
        model: Picklable model object.
        metadata: JSON-serializable metadata; created_utc, library_versions and model_sha256
            are added.
        out_dir: Target directory; created, and must not exist non-empty.

    Returns:
        `out_dir`.

    Raises:
        FileExistsError: If `out_dir` exists and is not empty.
        ValueError: If `metadata` already contains a reserved key.
    """
    reserved = {"created_utc", "library_versions", "model_sha256"} & set(metadata)
    if reserved:
        raise ValueError(f"metadata contains reserved keys: {sorted(reserved)}")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / _MODEL_FILE
    joblib.dump(model, model_path)
    meta = dict(metadata)
    meta["created_utc"] = datetime.now(UTC).isoformat()
    meta["library_versions"] = {"python": platform.python_version()} | {
        lib: version(lib) for lib in _LIBRARIES
    }
    meta["model_sha256"] = file_sha256(model_path)
    (out_dir / _META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True))
    logger.info("saved artifact to %s", out_dir)
    return out_dir


def load_artifact(out_dir: Path) -> tuple[Any, dict]:
    """Load a model and metadata, verifying the model checksum.

    Args:
        out_dir: Directory written by `save_artifact`.

    Returns:
        Tuple (model, metadata).

    Raises:
        ValueError: If the stored SHA-256 does not match model.joblib.
    """
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / _META_FILE).read_text())
    actual = file_sha256(out_dir / _MODEL_FILE)
    if actual != meta.get("model_sha256"):
        raise ValueError(
            f"model_sha256 mismatch in {out_dir}: {actual} != {meta.get('model_sha256')}"
        )
    return joblib.load(out_dir / _MODEL_FILE), meta
