# BalChat — from-scratch research prototype

BalChat is a new, small Balochi text-completion experiment. Its model weights and byte-level BPE tokenizer were initialized/trained without a pretrained checkpoint. The architecture uses the Llama implementation in Transformers; this is not an adaptation or renamed copy of the previous Qwen LoRA.

**This prototype does not provide verified natural Balochi chat, translation, reasoning or agentic tool use.** Those are project goals, not demonstrated capabilities. No claim of being the first Balochi model or matching ChatGPT, Gemini or Claude is made.

## Publication status

Training finished on the GPU machine. Both full checkpoints have been retrieved, matched against the GPU file hashes and passed CPU validation/state parity. The commercial checkpoint also passed offline inference checks. Hugging Face upload awaits an authenticated publishing session. `release.json` deliberately has no revision until uploaded bytes can be verified. The demo loader stops with a pending-checkpoint error. The earlier Qwen demo is still available; BalChat has not replaced it.

## What was built

- A byte BPE targeting 4096 tokens (3707 commercial, 4096 research) trained only on training documents, preserving Arabic/Latin Unicode, accents and whitespace.
- A decoder with six layers, width 256, MLP 768, eight query heads/four KV heads, tied embeddings and context 512: 5,670,912 commercial / 5,770,496 research full trainable parameters.
- Full FP32 AdamW training from random initialization on one existing TITAN X Pascal GPU.
- Document-group-separated train/validation/test inputs with receipt/hash verification. Every document token plus EOS is predicted once per pass; BOS supplies initial context. Gradient accumulation weights each microbatch by target-token count.
- Checkpoint selection by validation loss, separate test evaluation after selection, and saved-weight hash/reload parity checks.

| Experiment | Train records | Train word occurrences | Release |
|---|---:|---:|---|
| Commercial |16|15,818|Public model only|
| Mixed research |128|259,737|Private, noncommercial|

The mixed research set includes the commercial text. Across all 139 prepared records there are 281,089 word occurrences; validation/test text is not training data. Repeated epochs do not increase the amount of collected data. Native-reviewed training records and verified instruction pairs are both 0. The public training pool is predominantly Latin-script; byte encoding of Arabic is not proof of Arabic-script fluency.

## Reproduce

Python 3.12 and the pinned library versions are recommended. Install a Torch build appropriate to your hardware. Raw/prepared data is excluded from this public repo. `config/corpus.json` contains frozen receipts, not corpus text. The separate BalCorpus-Final data and its source terms are required to reproduce the original preparation.

```sh
python -m pip install -r requirements.txt
python -m pytest tests -q
# Prepared input must pass the frozen manifest/source/record/notices checks.
CUDA_VISIBLE_DEVICES=0 python train.py --prepared /path/to/prepared/commercial --purpose commercial --output runs/commercial --experimental
```

The default is five epochs, block 256, batch 1/accum 8, learning rate 0.0003, at most 1500 optimizer updates, and a soft 1200-second elapsed training-loop budget. Time is checked before each update, includes inter-epoch evaluation/checkpointing, and excludes tokenization, initial validation and final reload. A final evaluation/reload can overrun the limit. Validation early stopping follows two non-improving epochs. The original test partition is evaluated only after selecting the checkpoint.

`verify.py` can verify a saved experiment without retraining. It also repairs the old generic-tokenizer input-field metadata while asserting that learned tokenizer.json bytes remain unchanged. Model tensors are protected by the selected-state digest and loss reproduction.

## Recorded results

The commercial run completed 90 optimizer updates over five epochs. Validation NLL fell from 8.258177 to 5.727272 (perplexity 3859.053010 to 307.130297, 4788 predicted tokens). Saved weights reproduced the selected loss on the GPU host. The four diagnostic requests for greetings, translation and arithmetic produced punctuation or repetition instead of useful answers. These are observed failures; lower validation loss does not establish assistant quality.

The private research run completed 1350 updates over five epochs. Validation NLL fell from 8.373118 to 4.896609; test NLL after checkpoint selection was 4.462040 (22,333 predicted tokens). CPU reload reproduced its selected validation loss and model-state digest. Its Latin-script request produced Arabic-script output, and reasoning was not demonstrated. Its checkpoint and diagnostic text remain private.

The commercial reports in `reports/` include the completed GPU run and fresh CPU verification after repairing tokenizer input-field metadata. Model tensors and learned BPE bytes are unchanged. Their original `time_budget_scope` label overstated which evaluation was excluded; the actual timing behavior is described above. Perplexity is specific to each tokenizer and cannot be compared directly with Qwen or the research tokenizer. An offline integration check also loaded the exact commercial export through the demo's loader and generated continuations for Arabic and Latin prefixes; this checks software execution, not language quality or live Hub/Space deployment.

## Run the demo

```sh
python -m pip install -r requirements-space.txt
python app.py
```

`release.json` pins the public BalChat checkpoint and weight checksum. The loader rejects private-purpose or altered artifacts and does not execute custom remote model code. CPU is the default for this tiny model; optional ZeroGPU acceleration has provider quotas. The app performs text continuation, not a trained chat protocol, and does not save input text/history to disk. Hugging Face hosting has its own policies.

## Data and model terms

Public-source credits are in SOURCE_CREDITS.md. Commercial model text sources are CC BY 4.0; source attribution remains applicable. No private/noncommercial corpus or research weights are published with this code. Code is MIT; public scratch weights use their model card's Apache 2.0 terms. Private research material retains its separate source restrictions.

## Remaining work

The billion-scale assistant requested by the project owner is not delivered by this small experiment. It requires much more authorized text, reviewed conversational/parallel/task data, a funded training plan, native evaluation across scripts/dialects, and separate reasoning/tool-use tests. Repeating the current small corpus or generating synthetic text cannot be reported as billions of newly collected native words.
