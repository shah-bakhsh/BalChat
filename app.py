"""Small from-scratch BalChat demo; CPU default and optional free GPU."""
import os
import spaces
import torch
import gradio as gr

from inference import BalChat, validate_request

torch.set_num_threads(2)
cpu_model = BalChat("cpu")
# ZeroGPU CUDA emulation requires placement at module startup.
gpu_model = BalChat("cuda") if os.environ.get("SPACE_ID") else None


@spaces.GPU(duration=5)
def gpu_complete(prefix, max_tokens):
    return gpu_model.complete(prefix, max_tokens)


def complete(prefix, max_tokens, use_gpu=False):
    try:
        prefix, max_tokens = validate_request(prefix, max_tokens)
        if use_gpu:
            if gpu_model is None:
                raise ValueError("GPU acceleration is unavailable here. Leave it unchecked to use CPU.")
            return gpu_complete(prefix, max_tokens)
        return cpu_model.complete(prefix, max_tokens)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(title="BalChat — Scratch Prototype", analytics_enabled=False) as demo:
    gr.Markdown("# BalChat\nA small Balochi text-completion research prototype trained from random weights. "
                "**This is not a reliable chat assistant.** Arabic/Latin byte encoding is supported; natural "
                "Balochi, translation, reasoning and tool use are unverified. "
                "[Model and measured results](https://huggingface.co/shah-bakhsh/BalChat).")
    prefix = gr.Textbox(label="Text to continue", placeholder="Enter some Balochi text in Arabic or Latin script…", lines=3, max_length=2000)
    tokens = gr.Slider(16, 64, value=48, step=1, label="Maximum new tokens")
    use_gpu = gr.Checkbox(value=False, label="Use GPU acceleration", info="CPU works without GPU quota. Optional GPU usage has the provider's free limits.", visible=gpu_model is not None)
    generate = gr.Button("Generate continuation", variant="primary")
    output = gr.Textbox(label="Generated continuation — unreviewed", lines=4)
    generate.click(complete, [prefix, tokens, use_gpu], output, api_name="complete", concurrency_limit=1)
    gr.Markdown("Outputs may be incorrect, repetitive or empty. No verified instruction examples were used. This app does not save input text or chat histories to disk.")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch()
