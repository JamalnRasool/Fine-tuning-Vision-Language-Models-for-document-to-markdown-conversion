import gradio as gr
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image

# ── Load base model in 4-bit ──────────────────────────────────────────────────
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    quantization_config=quant_config,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "rek49/qwen2-vl-document-markdown",
    min_pixels=128*28*28,
    max_pixels=256*28*28
)

# ── Load your fine-tuned LoRA weights ────────────────────────────────────────
model = PeftModel.from_pretrained(base_model, "rek49/qwen2-vl-document-markdown")
model.eval()

# ── Inference function ────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return "⚠️ Please upload an image first."

    image = image.convert("RGB").resize((224, 224))

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": "Convert this document image to Markdown format."}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # strip everything before "assistant" turn
    if "assistant" in result.lower():
        result = result[result.lower().rfind("assistant") + len("assistant"):].strip()

    return result


# ── Gradio UI ─────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #e8e8f0 !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

#header {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2.5rem;
}

#header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #00d4ff, #7b2fff, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.6rem;
}

#header p {
    color: #888;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

.input-panel, .output-panel {
    background: #10101a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.5rem;
}

label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: #00d4ff !important;
    margin-bottom: 0.5rem !important;
}

button.primary {
    background: linear-gradient(135deg, #00d4ff, #7b2fff) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    padding: 0.75rem 2rem !important;
    color: #fff !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
    margin-top: 1rem !important;
}

button.primary:hover { opacity: 0.85 !important; }

textarea {
    background: #0a0a0f !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
    padding: 1rem !important;
}

.footer-note {
    text-align: center;
    margin-top: 2rem;
    font-size: 0.78rem;
    color: #444;
    font-family: 'Space Mono', monospace;
}
"""

with gr.Blocks(css=css, title="Doc → Markdown") as app:

    gr.HTML("""
    <div id="header">
        <h1>⚡ DOC → MARKDOWN</h1>
        <p>Fine-tuned Qwen2-VL · QLoRA · Document Understanding</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(elem_classes="input-panel"):
            image_input = gr.Image(
                type="pil",
                label="Upload Document Image"
            )
            submit_btn = gr.Button("Generate Markdown ›", variant="primary")

        with gr.Column(elem_classes="output-panel"):
            markdown_output = gr.Textbox(
                label="Generated Markdown",
                lines=20,
                placeholder="Your markdown will appear here..."
            )

    submit_btn.click(fn=predict, inputs=image_input, outputs=markdown_output)

    gr.HTML("""
    <div class="footer-note">
        model: rek49/qwen2-vl-document-markdown · base: Qwen2-VL-2B-Instruct
    </div>
    """)

app.launch()