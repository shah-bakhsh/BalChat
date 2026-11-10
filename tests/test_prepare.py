"""Engineering fixtures are not Balochi language data or actual sources."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import prepare, load_prepared


def test_commercial_export_rejects_nonempty_private_destination(tmp_path):
    corpus, receipt = fixture(tmp_path)
    output = tmp_path / "used-export"
    output.mkdir()
    private = output / "private-notice.txt"
    private.write_text("keep private", encoding="utf-8")
    with pytest.raises(ValueError, match="new|empty"):
        prepare(corpus, output, "commercial", receipt)
    assert private.read_text(encoding="utf-8") == "keep private"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, mutation=None):
    root = tmp_path / "corpus"
    sources = {
        "public": {"source_license": "CC-BY-4.0", "purpose_class": "commercial", "license_evidence_url": "https://example.org/public"},
        "nc": {"source_license": "CC-BY-NC-SA-4.0", "purpose_class": "noncommercial", "license_evidence_url": "https://example.org/nc"},
    }
    dump(root / "config/source_allowlist.json", sources)
    files = []
    record_receipts = {}
    for family in ("commercial", "noncommercial"):
        for split in ("train", "validation", "test"):
            text = "á a\u0301 \u0628\u200c\u06cc" if split == "train" else split + " fixture text"
            record = {
                "id": family + ":" + split, "source_id": "public" if family == "commercial" else "nc",
                "source_url": "https://example.org/text", "text_clean": text,
                "clean_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "split": split, "purpose_class": family, "human_reviewed": False,
                "source_license": "CC-BY-4.0" if family == "commercial" else "CC-BY-NC-SA-4.0",
                "license_evidence_url": "https://example.org/public" if family == "commercial" else "https://example.org/nc",
                "training_permission": "verified_subject_to_license", "redistribution_permission": "verified_subject_to_license",
                "recommended_for_review": True, "final_split_group_id": family + ":" + split,
                "parent_group": family + ":" + split, "attribution": "Test fixture author",
            }
            if mutation:
                mutation(record)
            record_receipts[record["id"]] = {
                **{key: record[key] for key in ("clean_sha256", "split", "purpose_class", "source_id", "source_license", "human_reviewed", "parent_group")},
                "group_id": record["final_split_group_id"],
            }
            path = root / "data/recommended" / family / (split + ".jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            files.append({"path": path.relative_to(root).as_posix(), "sha256": sha(path)})
    source = root / "config/source_allowlist.json"
    files.append({"path": "config/source_allowlist.json", "sha256": sha(source)})
    dump(root / "manifest.json", {"files": files})
    receipt = {"manifest_sha256": sha(root / "manifest.json"), "source_allowlist_sha256": sha(source), "record_receipts": record_receipts}
    return root, receipt


def test_commercial_export_preserves_unicode_and_original_holdouts(tmp_path):
    root, receipt = fixture(tmp_path)
    out = tmp_path / "export"
    report = prepare(root, out, "commercial", receipt)
    rows = load_prepared(out, "commercial", receipt)
    assert report["splits"]["train"]["records"] == 1
    assert rows["train"][0]["text"] == "á a\u0301 \u0628\u200c\u06cc"
    assert rows["train"][0]["id"] == "commercial:train"
    assert rows["validation"][0]["id"] == "commercial:validation"
    assert rows["test"][0]["id"] == "commercial:test"
    assert {r["source_license"] for v in rows.values() for r in v} == {"CC-BY-4.0"}
    assert all(r["human_reviewed"] is False for v in rows.values() for r in v)


def test_noncommercial_export_retains_both_original_license_families(tmp_path):
    root, receipt = fixture(tmp_path)
    out = tmp_path / "export"
    prepare(root, out, "noncommercial", receipt)
    rows = load_prepared(out, "noncommercial", receipt)
    assert len(rows["train"]) == 2
    assert {r["source_license"] for r in rows["train"]} == {"CC-BY-4.0", "CC-BY-NC-SA-4.0"}
    with pytest.raises(ValueError, match="purpose"):
        load_prepared(out, "commercial", receipt)


def test_nc_record_mislabeled_commercial_is_rejected(tmp_path):
    def mutate(r):
        if r["purpose_class"] == "commercial":
            r["source_id"] = "nc"
            r["source_license"] = "CC-BY-NC-SA-4.0"
            r["license_evidence_url"] = "https://example.org/nc"
    root, receipt = fixture(tmp_path, mutate)
    with pytest.raises(ValueError, match="purpose|license"):
        prepare(root, tmp_path / "export", "commercial", receipt)


@pytest.mark.parametrize("kind", ["manifest", "data", "allowlist"])
def test_tampered_input_fails_before_export(tmp_path, kind):
    root, receipt = fixture(tmp_path)
    path = {"manifest": "manifest.json", "data": "data/recommended/commercial/train.jsonl", "allowlist": "config/source_allowlist.json"}[kind]
    with (root / path).open("a", encoding="utf-8") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="hash|receipt"):
        prepare(root, tmp_path / "export", "commercial", receipt)
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("field", ["final_split_group_id", "parent_group"])
def test_shared_group_cannot_cross_original_partitions(tmp_path, field):
    root, receipt = fixture(tmp_path, lambda r: r.update({field: "shared"}))
    with pytest.raises(ValueError, match="group|partition"):
        prepare(root, tmp_path / "export", "commercial", receipt)


def test_bad_text_hash_cannot_enter_export(tmp_path):
    root, receipt = fixture(tmp_path, lambda r: r.update(clean_sha256="0" * 64))
    with pytest.raises(ValueError, match="text.*hash|hash.*text"):
        prepare(root, tmp_path / "export", "commercial", receipt)


def test_changed_prepared_text_is_not_trainable(tmp_path):
    root, receipt = fixture(tmp_path)
    out = tmp_path / "export"
    prepare(root, out, "commercial", receipt)
    with (out / "train.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"text":"injected"}\n')
    with pytest.raises(ValueError, match="hash"):
        load_prepared(out, "commercial", receipt)


def test_frozen_source_allowlist_supports_publisher_permission_variants(tmp_path):
    def mutate(r):
        r["training_permission"] = None if r["purpose_class"] == "commercial" else "publisher_declared_noncommercial_license_subject_to_terms"
        r["redistribution_permission"] = "not_certified_for_public_rehosting"
    root, receipt = fixture(tmp_path, mutate)
    report = prepare(root, tmp_path / "export", "noncommercial", receipt)
    assert report["splits"]["train"]["records"] == 2


def test_explicit_training_denial_is_not_overridden_by_license_label(tmp_path):
    root, receipt = fixture(tmp_path, lambda r: r.update(training_permission="denied"))
    with pytest.raises(ValueError, match="permission"):
        prepare(root, tmp_path / "export", "commercial", receipt)


def test_original_credit_notices_follow_export_and_are_verified(tmp_path):
    root, receipt = fixture(tmp_path)
    notice = root / "LICENSES.md"
    notice.write_text("Original fixture attribution notice", encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"].append({"path": "LICENSES.md", "sha256": sha(notice)})
    dump(root / "manifest.json", manifest)
    receipt["manifest_sha256"] = sha(root / "manifest.json")
    receipt["notice_files"] = {"LICENSES.md": {"sha256": sha(notice), "purpose_class": "commercial"}}
    out = tmp_path / "export"
    prepare(root, out, "commercial", receipt)
    assert (out / "notices/LICENSES.md").read_text() == "Original fixture attribution notice"
    (out / "notices/LICENSES.md").write_text("Changed credits", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_prepared(out, "commercial", receipt)


def test_rehashed_export_cannot_move_original_validation_record_into_training(tmp_path):
    root, receipt = fixture(tmp_path)
    out = tmp_path / "export"
    prepare(root, out, "commercial", receipt)
    train = json.loads((out / "train.jsonl").read_text(encoding="utf-8"))
    validation = json.loads((out / "validation.jsonl").read_text(encoding="utf-8"))
    train["split"], validation["split"] = "validation", "train"
    for split, row in (("train", validation), ("validation", train)):
        (out / (split + ".jsonl")).write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["train.jsonl"] = sha(out / "train.jsonl")
    manifest["files"]["validation.jsonl"] = sha(out / "validation.jsonl")
    dump(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="original.*receipt|receipt.*record"):
        load_prepared(out, "commercial", receipt)
