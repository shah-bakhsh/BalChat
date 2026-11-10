"""Small engineering fixtures; these are not native Balochi quality tests."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scratch import train_tokenizer, make_model, build_chunks, collate, token_weighted_loss, fit


def test_byte_tokenizer_preserves_unseen_unicode_and_whitespace():
    tokenizer = train_tokenizer(["training words only " * 30], 300)
    text = "Balóčí áéíóú\nءِ گ ء\u200c  ❤"
    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False), clean_up_tokenization_spaces=False) == text
    assert tokenizer.backend_tokenizer.normalizer is None
    assert [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id] == [0, 1, 2, 3]


def test_tokenizer_does_not_send_unsupported_segment_ids_to_decoder():
    tokenizer = train_tokenizer(["training text"], 270)
    assert set(tokenizer("test")) == {"input_ids", "attention_mask"}


def test_legacy_input_field_repair_survives_save_and_fresh_reload(tmp_path):
    from transformers import AutoTokenizer
    from verify import repair_tokenizer
    tokenizer = train_tokenizer(["old checkpoint fixture"], 280)
    tokenizer.init_kwargs.pop("model_input_names", None)
    tokenizer.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    tokenizer.save_pretrained(tmp_path)
    assert "token_type_ids" in AutoTokenizer.from_pretrained(tmp_path)("test")
    old_bytes = (tmp_path / "tokenizer.json").read_bytes()
    repair_tokenizer(tmp_path)
    assert set(AutoTokenizer.from_pretrained(tmp_path)("test")) == {"input_ids", "attention_mask"}
    assert (tmp_path / "tokenizer.json").read_bytes() == old_bytes


def test_chunks_count_every_document_token_and_eos_once():
    tokenizer = train_tokenizer(["a b c " * 5], 270)
    text = "a b c d e f g h i"
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = build_chunks([{"text": text}], tokenizer, 4)
    assert chunks[0][0] == tokenizer.bos_token_id
    assert [v for chunk in chunks for v in chunk[1:]] == ids + [tokenizer.eos_token_id]
    assert all(2 <= len(c) <= 4 for c in chunks)


def test_padding_never_masks_real_eos_or_changes_target_count():
    batch = collate([[1, 20, 2], [1, 2]], 0)
    assert batch["labels"].tolist() == [[1, 20, 2], [1, 2, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_new_model_has_no_pretrained_reference_and_all_parameters_train():
    tokenizer = train_tokenizer(["sample only"], 270)
    model = make_model(tokenizer, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)
    assert model.config._name_or_path == ""
    assert all(p.requires_grad for p in model.parameters())
    assert model.config.vocab_size == len(tokenizer)
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()


@pytest.mark.parametrize("block", [0, 1])
def test_invalid_causal_block_is_rejected(block):
    tokenizer = train_tokenizer(["a"], 260)
    with pytest.raises(ValueError):
        build_chunks([{"text": "a"}], tokenizer, block)


def test_full_model_can_train_save_reload_and_reproduce_validation(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2)
    torch.manual_seed(42)
    tokenizer = train_tokenizer(["one two three " * 3], 280)
    model = make_model(tokenizer, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, attention_dropout=0)
    chunks = build_chunks([{"text": "one two three " * 3}], tokenizer, 16)
    output = tmp_path / "result"
    report = fit(model, chunks, chunks, output, epochs=4, accumulation=2, learning_rate=0.01, max_steps=4)
    assert report["optimizer_steps"] == 4
    assert report["trainable_parameters"] == report["total_parameters"]
    assert report["initial_weights_sha256"] != report["selected_weights_sha256"]
    assert report["initialization"] == "random_no_pretrained_weights"
    reloaded = AutoModelForCausalLM.from_pretrained(output / "model", local_files_only=True)
    measured = token_weighted_loss(reloaded, chunks)
    assert measured["token_weighted_nll"] == pytest.approx(report["selected_validation"]["token_weighted_nll"], abs=1e-6)
    assert measured["predicted_tokens"] == sum(len(c) - 1 for c in chunks)
    assert report["seen_training_target_tokens"] > 0


def test_output_directory_is_not_silently_reused(tmp_path):
    existing = tmp_path / "result"
    existing.mkdir()
    (existing / "owner-file").write_text("keep")
    with pytest.raises(ValueError, match="empty|new"):
        fit(None, [[1, 2]], [[1, 2]], existing)
    assert (existing / "owner-file").read_text() == "keep"
