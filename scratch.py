"""BalChat random-initialized causal language model research helpers."""
import hashlib
import json
import math
from pathlib import Path
import random
import time


def train_tokenizer(texts, vocab_size=4096):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    if vocab_size < 260:
        raise ValueError("Vocabulary must cover all 256 bytes and four special tokens")
    backend = Tokenizer(models.BPE(unk_token="<|unk|>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(texts, trainer=trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    ))
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<|pad|>", bos_token="<|bos|>",
        eos_token="<|eos|>", unk_token="<|unk|>",
        model_input_names=["input_ids", "attention_mask"],
        model_max_length=512, clean_up_tokenization_spaces=False,
    )


def make_model(tokenizer, **overrides):
    from transformers import LlamaConfig, LlamaForCausalLM
    options = dict(
        vocab_size=len(tokenizer), hidden_size=256, intermediate_size=768,
        num_hidden_layers=6, num_attention_heads=8, num_key_value_heads=4,
        max_position_embeddings=512, tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, attention_dropout=0.1,
        rms_norm_eps=1e-5, rope_theta=10000.0, attn_implementation="sdpa",
    )
    options.update(overrides)
    # Constructing from a configuration initializes every weight randomly.
    # No from_pretrained() call or inherited checkpoint is used here.
    return LlamaForCausalLM(LlamaConfig(**options))


def build_chunks(rows, tokenizer, block_size=256):
    if not 2 <= block_size <= 512:
        raise ValueError("Block size must be between 2 and 512")
    result = []
    for row in rows:
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        sequence = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
        result.extend(sequence[i:i + block_size] for i in range(0, len(sequence) - 1, block_size - 1))
    return result


def collate(chunks, pad_id=0):
    import torch
    width = max(len(row) for row in chunks)
    return {
        "input_ids": torch.tensor([row + [pad_id] * (width - len(row)) for row in chunks]),
        "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in chunks]),
        "labels": torch.tensor([row + [-100] * (width - len(row)) for row in chunks]),
    }


def token_weighted_loss(model, chunks):
    import torch
    if not chunks:
        raise ValueError("Nonempty evaluation text is required")
    device = next(model.parameters()).device
    prior = model.training
    model.eval()
    total, targets = 0.0, 0
    with torch.inference_mode():
        for chunk in chunks:
            batch = {k: v.to(device) for k, v in collate([chunk], model.config.pad_token_id).items()}
            count = len(chunk) - 1
            loss = float(model(**batch, use_cache=False).loss)
            if not math.isfinite(loss):
                raise ValueError("Nonfinite evaluation loss")
            total += loss * count
            targets += count
    model.train(prior)
    nll = total / targets
    return {"token_weighted_nll": nll, "perplexity": math.exp(nll), "predicted_tokens": targets}


def state_hash(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode() + b"\0")
        result.update(str(tensor.shape).encode() + b"\0")
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def fit(model, train_chunks, validation_chunks, output, **options):
    import torch
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    if not train_chunks or not validation_chunks:
        raise ValueError("Training and validation text are required")
    epochs = options.get("epochs", 5)
    accumulation = options.get("accumulation", 8)
    max_steps = options.get("max_steps", 1500)
    learning_rate = options.get("learning_rate", 3e-4)
    if epochs < 1 or accumulation < 1 or max_steps < 1 or learning_rate <= 0:
        raise ValueError("Training options must be positive")
    output.mkdir(parents=True, exist_ok=True)
    seed = options.get("seed", 42)
    torch.manual_seed(seed)
    before = token_weighted_loss(model, validation_chunks)
    report = {
        "initialization": "random_no_pretrained_weights", "seed": seed,
        "initial_weights_sha256": state_hash(model),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_validation": before, "selected_validation": before,
        "optimizer_steps": 0, "selected_step": 0, "seen_training_target_tokens": 0,
        "one_pass_training_target_tokens": sum(len(c) - 1 for c in train_chunks),
        "history": [], "native_quality_verified": False,
        "time_budget_scope": "elapsed_training_loop_includes_inter_epoch_eval_excludes_initial_eval_and_final_reload",
    }
    if report["total_parameters"] != report["trainable_parameters"]:
        raise ValueError("Scratch experiment requires full-parameter training")
    model.save_pretrained(output / "model", safe_serialization=True)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": 0.1},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ], lr=learning_rate, betas=(0.9, 0.95))
    total_steps = min(max_steps, math.ceil(len(train_chunks) / accumulation) * epochs)
    warmup = max(1, min(20, total_steps // 10))
    started, stale_epochs = time.monotonic(), 0
    stop_reason = "epochs_complete"
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_chunks)))
        random.Random(seed + epoch).shuffle(order)
        model.train()
        for offset in range(0, len(order), accumulation):
            if report["optimizer_steps"] >= max_steps:
                stop_reason = "step_limit"
                break
            if time.monotonic() - started >= options.get("max_seconds", 1200):
                stop_reason = "time_limit"
                break
            group = [train_chunks[i] for i in order[offset:offset + accumulation]]
            group_targets = sum(len(c) - 1 for c in group)
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for chunk in group:
                batch = {k: v.to(device) for k, v in collate([chunk], model.config.pad_token_id).items()}
                loss = model(**batch, use_cache=False).loss
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite training loss")
                weighted = loss * ((len(chunk) - 1) / group_targets)
                weighted.backward()
                group_loss += float(weighted.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            step = report["optimizer_steps"]
            if step < warmup:
                factor = (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, total_steps - warmup)
                factor = 0.1 + 0.45 * (1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate * factor
            optimizer.step()
            report["optimizer_steps"] += 1
            report["seen_training_target_tokens"] += group_targets
            if report["optimizer_steps"] % 25 == 0:
                print(json.dumps({"epoch": epoch, "step": report["optimizer_steps"], "training_nll": group_loss, "seconds": round(time.monotonic() - started, 2)}), flush=True)
        measured = token_weighted_loss(model, validation_chunks)
        report["history"].append({"epoch": epoch, "step": report["optimizer_steps"], "validation": measured})
        print(json.dumps(report["history"][-1]), flush=True)
        if measured["token_weighted_nll"] < report["selected_validation"]["token_weighted_nll"]:
            report["selected_validation"] = measured
            report["selected_step"] = report["optimizer_steps"]
            model.save_pretrained(output / "model", safe_serialization=True)
            stale_epochs = 0
        else:
            stale_epochs += 1
        (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if stop_reason != "epochs_complete" or report["optimizer_steps"] >= max_steps:
            stop_reason = "step_limit" if report["optimizer_steps"] >= max_steps else stop_reason
            break
        if stale_epochs >= 2:
            stop_reason = "validation_early_stop"
            break
    from transformers import AutoModelForCausalLM
    selected = AutoModelForCausalLM.from_pretrained(output / "model", local_files_only=True)
    report["selected_weights_sha256"] = state_hash(selected)
    report["stop_reason"] = stop_reason
    report["training_seconds"] = round(time.monotonic() - started, 3)
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
