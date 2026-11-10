"""Pinned public BalChat text-completion loader."""
import hashlib
import json
import math
from pathlib import Path
import threading
import re


def validate_pin(release):
    if not isinstance(release.get("revision"), str) or not re.fullmatch(r"[a-f0-9]{40}", release["revision"]):
        raise ValueError("Checkpoint upload is pending; an immutable Hub revision is required.")
    if release.get("model_id") != "shah-bakhsh/BalChat" or not re.fullmatch(r"[a-f0-9]{64}", release.get("weights_sha256", "")):
        raise ValueError("Invalid public release identity or checksum.")
    return release


def validate_request(prefix, max_tokens):
    if not isinstance(prefix, str) or not prefix.strip() or len(prefix) > 2000:
        raise ValueError("Enter 1 to 2000 characters of text.")
    if type(max_tokens) not in {int, float} or not math.isfinite(max_tokens) or not float(max_tokens).is_integer() or not 16 <= max_tokens <= 64:
        raise ValueError("Maximum tokens must be an integer from 16 to 64.")
    return prefix, int(max_tokens)


def validate_release(folder, expected_weights_hash):
    folder = Path(folder).resolve()
    meta = json.loads((folder / "release-metadata.json").read_text(encoding="utf-8"))
    if meta.get("purpose") != "commercial":
        raise ValueError("The public demo requires the commercial model.")
    if meta.get("initialization") != "random_no_pretrained_weights" or meta.get("pretrained_weights_used") is not False:
        raise ValueError("This is not the declared from-scratch release.")
    files = meta.get("files", {})
    if not {"model.safetensors", "tokenizer.json", "config.json"}.issubset(files):
        raise ValueError("Incomplete model manifest.")
    if files["model.safetensors"] != expected_weights_hash:
        raise ValueError("Model hash differs from the pinned release.")
    for name, digest in files.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError("Invalid model file path.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Model file hash mismatch: " + name)
    if json.loads((folder / "config.json").read_text())["model_type"] != "llama":
        raise ValueError("Unexpected architecture.")
    return meta


class BalChat:
    def __init__(self, device="cpu"):
        release = validate_pin(json.loads(Path(__file__).with_name("release.json").read_text(encoding="utf-8")))
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        folder = snapshot_download(release["model_id"], revision=release["revision"], token=False,
            local_dir=Path.home() / ".cache" / "balchat" / release["revision"],
            allow_patterns=["*.json", "model.safetensors"])
        validate_release(folder, release["weights_sha256"])
        self.tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=False)
        if set(self.tokenizer("check")) != {"input_ids", "attention_mask"}:
            raise ValueError("Tokenizer emits unsupported model inputs.")
        self.model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=False,
            torch_dtype=torch.float32, attn_implementation="sdpa").to(device).eval()
        self.device, self.lock = device, threading.Lock()

    def complete(self, prefix, max_tokens=48):
        import torch
        prefix, limit = validate_request(prefix, max_tokens)
        ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prefix, add_special_tokens=False)
        if len(ids) > 256:
            raise ValueError("Please shorten the text to at most 255 tokenizer tokens.")
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        with self.lock, torch.inference_mode():
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=limit, do_sample=False, repetition_penalty=1.1, max_time=5,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True)
        return self.tokenizer.decode(output[0, len(ids):], skip_special_tokens=True)
