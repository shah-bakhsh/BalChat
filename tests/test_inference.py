import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import validate_request, validate_release


@pytest.mark.parametrize("revision", [None, "main", "short"])
def test_unpublished_or_floating_checkpoint_cannot_be_loaded(revision):
    from inference import validate_pin
    with pytest.raises(ValueError, match="pending|immutable"):
        validate_pin({"model_id": "shah-bakhsh/BalChat", "revision": revision, "weights_sha256": "a" * 64})


@pytest.mark.parametrize("prefix,tokens", [("", 32), ("x" * 2001, 32), ("a", True), ("a", 65), ("a", 32.5)])
def test_invalid_requests_are_rejected_before_allocating_gpu(prefix, tokens):
    with pytest.raises(ValueError):
        validate_request(prefix, tokens)


def test_valid_prefix_preserves_orthography():
    text = "á گ\u200cءِ\n "
    assert validate_request(text, 32) == (text, 32)


def fixture(folder, purpose="commercial"):
    files = {"model.safetensors": b"test-only engineering fixture", "tokenizer.json": b"{}", "config.json": b'{"model_type":"llama"}'}
    for name, data in files.items():
        (folder / name).write_bytes(data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    metadata = {"purpose": purpose, "initialization": "random_no_pretrained_weights", "pretrained_weights_used": False, "files": hashes}
    (folder / "release-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return hashes["model.safetensors"]


def test_private_model_is_rejected_by_public_demo(tmp_path):
    expected = fixture(tmp_path, "noncommercial")
    with pytest.raises(ValueError, match="commercial"):
        validate_release(tmp_path, expected)


def test_rehashing_local_manifest_cannot_replace_pinned_model(tmp_path):
    expected = fixture(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"different weights")
    meta = json.loads((tmp_path / "release-metadata.json").read_text())
    meta["files"]["model.safetensors"] = hashlib.sha256(b"different weights").hexdigest()
    (tmp_path / "release-metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="pinned|hash"):
        validate_release(tmp_path, expected)


def test_exact_public_release_is_accepted(tmp_path):
    expected = fixture(tmp_path)
    assert validate_release(tmp_path, expected)["purpose"] == "commercial"
