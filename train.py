"""Bounded full-parameter BalChat scratch experiment on receipt-bound text."""
import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import unicodedata

from prepare import ROOT, load_prepared, read_json, sha256, write_json
from scratch import train_tokenizer, make_model, build_chunks, token_weighted_loss, state_hash, fit


def script_counts(text):
    result = {"Arabic": 0, "Latin": 0}
    for char in text:
        if not unicodedata.category(char).startswith("L"):
            continue
        name = unicodedata.name(char, "")
        for script in result:
            if script.upper() in name:
                result[script] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--purpose", choices=["commercial", "noncommercial"], required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--max-seconds", type=int, default=1200, help="Soft elapsed training-loop budget; includes inter-epoch evaluation, excludes initial evaluation and final reload")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if not args.experimental:
        parser.error("Native quality is unreviewed. --experimental is required.")
    if min(args.epochs, args.max_steps, args.max_seconds) < 1:
        parser.error("Epoch, step and time limits must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new empty output directory")
    rows = load_prepared(args.prepared, args.purpose, read_json(ROOT / "config/corpus.json"))
    import torch
    from transformers import AutoModelForCausalLM, set_seed
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        parser.error("Select exactly one available GPU using CUDA_VISIBLE_DEVICES")
    set_seed(42)
    tokenizer = train_tokenizer((r["text"] for r in rows["train"]))
    model = make_model(tokenizer).to(args.device)
    summary = {
        "name": "BalChat", "version": "scratch-prototype-v1", "purpose": args.purpose,
        "visibility": "public" if args.purpose == "commercial" else "private",
        "prepared_manifest_sha256": sha256(args.prepared / "manifest.json"),
        "source_licenses": sorted({r["source_license"] for r in rows["train"]}),
        "partitions": {split: {"records": len(items), "word_occurrences": sum(len(r["text"].split()) for r in items)} for split, items in rows.items()},
        "native_reviewed_training_records": sum(r.get("human_reviewed") is True for r in rows["train"]),
        "verified_instruction_pairs": 0, "train_script_letters": script_counts("\n".join(r["text"] for r in rows["train"])),
        "tokenizer_trained_on": "train_only", "pretrained_weights_used": False,
        "vocabulary_size": len(tokenizer), "architecture": model.config.to_dict(),
        "hardware": {"device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None},
        "environment": {"python": platform.python_version(), **{k: importlib.metadata.version(k) for k in ["torch", "transformers", "tokenizers"]}},
        "capabilities": {key: "NOT_ESTABLISHED" for key in ["native_balochi_chat", "translation", "reasoning", "agentic_tool_use"]},
        "options": {"epochs": args.epochs, "max_steps": args.max_steps, "max_seconds": args.max_seconds, "block_size": 256, "accumulation": 8, "learning_rate": 0.0003},
    }
    train_chunks = build_chunks(rows["train"], tokenizer)
    validation_chunks = build_chunks(rows["validation"], tokenizer)
    report = fit(model, train_chunks, validation_chunks, args.output,
                 epochs=args.epochs, max_steps=args.max_steps, max_seconds=args.max_seconds)
    tokenizer.save_pretrained(args.output / "model")
    write_json(args.output / "data-summary.json", summary)
    if report["selected_step"] == 0 or report["selected_weights_sha256"] == report["initial_weights_sha256"]:
        raise RuntimeError("No trained checkpoint improved validation; do not publish as a trained model")
    del model
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    from verify import verify_result
    verify_result(args.output, args.prepared, args.purpose, args.device)



if __name__ == "__main__":
    main()
