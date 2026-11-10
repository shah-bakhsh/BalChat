"""Prepare the frozen recommended corpus for one declared use purpose (stdlib)."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
PURPOSES = ("commercial", "noncommercial")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA256 hash/receipt mismatch: {path}")


def check_record(record, split, family, allowlist, prepared=False):
    source = allowlist.get(record["source_id"])
    if not source:
        raise ValueError("Unknown source license/purpose")
    for key in ("source_license", "license_evidence_url", "purpose_class"):
        if record.get(key) != source[key]:
            raise ValueError(f"Source license/purpose evidence mismatch: {record['id']} / {key}")
    if record["purpose_class"] != family:
        raise ValueError("Record purpose differs from file family")
    if family == "commercial" and record["source_license"] not in {"CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-2.0-FR"}:
        raise ValueError("Unapproved commercial license")
    if record.get("split") != split:
        raise ValueError("Record partition differs from file partition")
    text = record["text" if prepared else "text_clean"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty training text")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != record["clean_sha256"]:
        raise ValueError("Training text hash mismatch")
    allowed = {None, "verified_subject_to_license", "publisher_declared_noncommercial_license_subject_to_terms"}
    if record.get("training_permission") not in allowed or (not prepared and not record.get("recommended_for_review")):
        raise ValueError("Source lacks recommended/permission evidence")
    if record.get("training_permission") == "publisher_declared_noncommercial_license_subject_to_terms" and family != "noncommercial":
        raise ValueError("Noncommercial training permission in commercial file")


def check_groups(rows):
    ids = set()
    partitions = {}
    for split, records in rows.items():
        for record in records:
            if record["id"] in ids:
                raise ValueError("Duplicate record id")
            ids.add(record["id"])
            for key in ("group_id", "parent_group"):
                group = record.get(key)
                if not group:
                    raise ValueError("Missing source group")
                name = (key, group)
                if partitions.setdefault(name, split) != split:
                    raise ValueError("Source group crosses partitions")


def check_original_records(rows, purpose, receipt):
    expected = receipt.get("record_receipts")
    if not expected:
        raise ValueError("Missing original record receipts")
    families = {"commercial"} if purpose == "commercial" else set(PURPOSES)
    expected_ids = {key for key, info in expected.items() if info["purpose_class"] in families}
    actual_ids = {r["id"] for records in rows.values() for r in records}
    if actual_ids != expected_ids:
        raise ValueError("Export record membership differs from original corpus receipt")
    for records in rows.values():
        for r in records:
            for key, value in expected[r["id"]].items():
                if key not in r or r[key] != value:
                    raise ValueError(f"Export differs from original record receipt: {r['id']} / {key}")


def prepare(corpus_root, out_dir, purpose, receipt):
    corpus_root, out_dir = Path(corpus_root).resolve(), Path(out_dir).resolve()
    if purpose not in PURPOSES:
        raise ValueError("Unknown purpose")
    if out_dir == corpus_root or corpus_root in out_dir.parents:
        raise ValueError("Output must stay outside the frozen corpus")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Use a new empty output directory for each purpose")
    require_hash(corpus_root / "manifest.json", receipt["manifest_sha256"])
    manifest = read_json(corpus_root / "manifest.json")
    files = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    allow_path = corpus_root / "config/source_allowlist.json"
    require_hash(allow_path, receipt["source_allowlist_sha256"])
    require_hash(allow_path, files["config/source_allowlist.json"])
    allowlist = read_json(allow_path)
    rows = {split: [] for split in SPLITS}
    families = ["commercial"] if purpose == "commercial" else list(PURPOSES)
    notices = {}
    for relative, info in receipt.get("notice_files", {}).items():
        if info["purpose_class"] in families:
            require_hash(corpus_root / relative, info["sha256"])
            require_hash(corpus_root / relative, files[relative])
            notices["notices/" + relative] = (corpus_root / relative).read_bytes()
    for family in families:
        for split in SPLITS:
            relative = f"data/recommended/{family}/{split}.jsonl"
            path = corpus_root / relative
            require_hash(path, files[relative])
            for r in read_jsonl(path):
                check_record(r, split, family, allowlist)
                rows[split].append({
                    "id": r["id"], "text": r["text_clean"], "clean_sha256": r["clean_sha256"],
                    "source_id": r["source_id"], "source_url": r["source_url"],
                    "source_license": r["source_license"], "license_evidence_url": r["license_evidence_url"],
                    "purpose_class": r["purpose_class"], "attribution": r.get("attribution"),
                    "training_permission": r.get("training_permission"), "redistribution_permission": r.get("redistribution_permission"),
                    "split": split, "group_id": r["final_split_group_id"], "parent_group": r["parent_group"],
                    "human_reviewed": r["human_reviewed"], "script": r.get("script", "unknown"),
                    "dialect": r.get("dialect", "unknown"),
                    "parent_file": r.get("parent_file"), "raw_file_sha256": r.get("raw_file_sha256"),
                    "source_offsets_codepoints": r.get("source_offsets_codepoints"),
                })
    check_groups(rows)
    check_original_records(rows, purpose, receipt)
    # Validate everything before creating output, so failures leave no partial export.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source_allowlist.json").write_bytes(allow_path.read_bytes())
    for name, payload in notices.items():
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    for split, records in rows.items():
        (out_dir / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    report = {
        "status": "experimental_candidates_NOT_native_verified_NOT_instruction_data",
        "purpose": purpose, "corpus_receipt": receipt,
        "splits": {s: {"records": len(rs), "whitespace_words": sum(len(r["text"].split()) for r in rs),
                        "native_reviewed_records": sum(r["human_reviewed"] is True for r in rs)} for s, rs in rows.items()},
        "files": {name: sha256(out_dir / name) for name in ["source_allowlist.json"] + [s + ".jsonl" for s in SPLITS] + list(notices)},
        "source_licenses": sorted({r["source_license"] for rs in rows.values() for r in rs}),
        "instructions_verified": 0, "native_generative_evaluation_verified": 0,
    }
    write_json(out_dir / "manifest.json", report)
    return report


def load_prepared(root, purpose, receipt):
    root = Path(root)
    report = read_json(root / "manifest.json")
    if purpose not in PURPOSES or report["purpose"] != purpose:
        raise ValueError("Prepared dataset purpose mismatch")
    if report["corpus_receipt"] != receipt:
        raise ValueError("Prepared dataset corpus receipt mismatch")
    require_hash(root / "source_allowlist.json", receipt["source_allowlist_sha256"])
    for name in ["source_allowlist.json"] + [s + ".jsonl" for s in SPLITS]:
        require_hash(root / name, report["files"][name])
    families = ["commercial"] if purpose == "commercial" else list(PURPOSES)
    for name, info in receipt.get("notice_files", {}).items():
        if info["purpose_class"] in families:
            require_hash(root / "notices" / name, info["sha256"])
    allowlist = read_json(root / "source_allowlist.json")
    rows = {s: read_jsonl(root / f"{s}.jsonl") for s in SPLITS}
    for split, records in rows.items():
        for r in records:
            if purpose == "commercial" and r["purpose_class"] != "commercial":
                raise ValueError("NC text cannot enter a commercial experiment")
            check_record(r, split, r["purpose_class"], allowlist, prepared=True)
    check_groups(rows)
    check_original_records(rows, purpose, receipt)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT.parent / "BalCorpus-Final")
    parser.add_argument("--purpose", choices=PURPOSES, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "prepared" / args.purpose
    report = prepare(args.corpus, output, args.purpose, read_json(ROOT / "config/corpus.json"))
    print(json.dumps({"purpose": report["purpose"], "splits": report["splits"]}, indent=2))
