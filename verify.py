"""Verify an already trained BalChat checkpoint without retraining it."""
import argparse
from pathlib import Path
import json

from prepare import ROOT, load_prepared, read_json, sha256, write_json
from scratch import build_chunks, token_weighted_loss, state_hash
from train import script_counts


def repair_tokenizer(folder):
    from transformers import AutoTokenizer
    folder = Path(folder)
    original_bpe = (folder / "tokenizer.json").read_bytes()
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    names = ["input_ids", "attention_mask"]
    tokenizer.model_input_names = names
    tokenizer.init_kwargs["model_input_names"] = names
    tokenizer.save_pretrained(folder)
    reloaded = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    if set(reloaded("check")) != set(names) or (folder / "tokenizer.json").read_bytes() != original_bpe:
        raise ValueError("Tokenizer metadata repair did not preserve the learned byte BPE")
    return reloaded


def verify_result(output, prepared, purpose, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    output, prepared = Path(output), Path(prepared)
    rows = load_prepared(prepared, purpose, read_json(ROOT / "config/corpus.json"))
    summary = read_json(output / "data-summary.json")
    report = read_json(output / "training.json")
    if summary["purpose"] != purpose or summary["prepared_manifest_sha256"] != sha256(prepared / "manifest.json"):
        raise ValueError("Saved model purpose or input receipt differs")
    if report["selected_step"] == 0 or report["selected_weights_sha256"] == report["initial_weights_sha256"]:
        raise ValueError("Selected checkpoint has no trained improvement")
    # Recovery for the first checkpoint: only input-field metadata changes.
    # Learned BPE vocabulary, IDs and model tensors stay exactly the same.
    tokenizer = repair_tokenizer(output / "model")
    reloaded = AutoModelForCausalLM.from_pretrained(output / "model", local_files_only=True, torch_dtype=torch.float32).to(device).eval()
    validation_chunks = build_chunks(rows["validation"], tokenizer)
    actual = token_weighted_loss(reloaded, validation_chunks)
    if abs(actual["token_weighted_nll"] - report["selected_validation"]["token_weighted_nll"]) > 1e-5:
        raise ValueError("Reload validation differs from selected checkpoint")
    if state_hash(reloaded) != report["selected_weights_sha256"] or not all(bool(torch.isfinite(p).all()) for p in reloaded.parameters()):
        raise ValueError("Saved weights failed integrity/finite checks")
    # Test partitions are first tokenized and evaluated only after checkpoint selection.
    test = token_weighted_loss(reloaded, build_chunks(rows["test"], tokenizer)) if rows["test"] else None
    diagnostics = []
    prompts = [
        ("arabic_greeting", "Write one short greeting in Balochi using Arabic script.", "Arabic"),
        ("latin_greeting", "Write one short greeting in Balochi using Latin script.", "Latin"),
        ("english_translation", "Translate into Balochi: Hello, how are you?", None),
        ("reasoning", "Answer in Balochi: I have two books and receive three more. How many books do I have?", None),
    ]
    for name, prompt, requested in prompts:
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = reloaded.generate(**encoded, max_new_tokens=48, do_sample=False,
                repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, use_cache=True)
        text = tokenizer.decode(generated[0, encoded.input_ids.shape[1]:], skip_special_tokens=True)
        letters = script_counts(text)
        diagnostics.append({"id": name, "prompt": prompt, "output": text, "requested_script": requested,
            "letter_counts": letters, "native_correctness": None, "synthetic_unreviewed_not_training_data": True})
    verification = {"reloaded_validation": actual, "reload_matches_selected": True,
        "test_after_selection": test, "test_used_for_training_or_selection": False,
        "all_weights_finite": True, "model_state_sha256": state_hash(reloaded),
        "file_sha256": {p.name: sha256(p) for p in (output / "model").iterdir() if p.is_file()},
        "diagnostic_generations": diagnostics, "native_quality_established": False}
    write_json(output / "verification.json", verification)

    print(json.dumps({"verified": True, "purpose": purpose, "parameters": report["total_parameters"],
        "steps": report["optimizer_steps"], "selected_validation": report["selected_validation"]}), flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--purpose", choices=["commercial", "noncommercial"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args=parser.parse_args()
    verify_result(args.output,args.prepared,args.purpose,args.device)


if __name__ == "__main__":
    main()
