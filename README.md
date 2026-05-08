#  Document-to-Markdown Generation with Qwen2-VL + QLoRA

Fine-tuning a Vision Language Model to convert scanned document images into clean, structured Markdown — complete with LaTeX equations and proper headings.

> **Authors:** [Jamal Rasool] · [Umar Zahoor](https://github.com/umar-zahoor)

---

##  Overview

This project fine-tunes **Qwen2-VL-2B-Instruct**, a 2-billion parameter multimodal model, using **QLoRA** (Quantized Low-Rank Adaptation) to perform document image → Markdown conversion. The entire training pipeline runs on **free Kaggle T4 x2 GPUs**, making large-scale VLM fine-tuning accessible without expensive hardware.

---

##  Features

-  Converts scanned document images to structured Markdown
-  Preserves LaTeX equations and mathematical notation
-  Maintains proper heading hierarchy and document structure
-  Trained with only ~5M / 2B parameters via LoRA adapters
-  Fully reproducible on free Kaggle GPUs

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| `PyTorch` | Deep learning framework |
| `HuggingFace Transformers` | Model loading & training |
| `PEFT` | LoRA adapter management |
| `BitsAndBytes` | 4-bit quantization (QLoRA) |
| `Gradio` | Interactive demo app |
| `Kaggle (T4 x2)` | Training infrastructure |

---

## Methodology

### 1. Model Loading
Qwen2-VL-2B-Instruct is loaded in **4-bit quantized form** using BitsAndBytes, drastically reducing GPU memory requirements while preserving model quality.

### 2. QLoRA Fine-Tuning
Rather than updating all 2B parameters, LoRA adapters are injected into the model's attention layers — training only **~5 million parameters** (~0.25% of total).

### 3. Dataset
Trained on the **[Nougat Document Dataset](https://huggingface.co/datasets/facebook/nougat)** from Kaggle, which contains paired document images and their Markdown representations.

### 4. Deployment
The fine-tuned model is deployed as a live **Gradio app on HuggingFace Spaces** for easy inference.

---

##  Getting Started

### Prerequisites

```bash
pip install torch transformers peft bitsandbytes gradio accelerate
```

### Clone the Repo

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Training

```bash
python train.py
```

>  You can also run the training notebook directly on Kaggle using T4 x2 GPUs for free.

### Inference

```bash
python inference.py --image path/to/document.png
```

### Run Gradio App Locally

```bash
python app.py
```

---

##  Results

| Metric | Value |
|--------|-------|
| Trainable Parameters | ~5M / 2B |
| Quantization | 4-bit (NF4) |
| Training Hardware | Kaggle T4 x2 (Free) |
| Output Format | Markdown + LaTeX |

---

##  Live Demo

 **[Try it on HuggingFace Spaces](https://huggingface.co/spaces/rek49/genai_assignmnet_5)**

---

##  Project Structure

```
├── train.py              # Training script
├── inference.py          # Run inference on a document image
├── app.py                # Gradio demo app
├── config.py             # Training hyperparameters & config
├── dataset.py            # Dataset loading & preprocessing
├── requirements.txt      # Python dependencies
└── README.md
```

---

##  Key Takeaway

> QLoRA makes fine-tuning billion-parameter models accessible to everyone — no expensive hardware needed. We ran the entire training pipeline on a **free Kaggle GPU**.

---

##  Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

##  License

[MIT](LICENSE)

---

##  Acknowledgements

- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) by Alibaba Cloud
- [Nougat](https://github.com/facebookresearch/nougat) by Meta Research
- [PEFT](https://github.com/huggingface/peft) by HuggingFace
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) by Tim Dettmers

- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) by Alibaba Cloud
- [Nougat](https://github.com/facebookresearch/nougat) by Meta Research
- [PEFT](https://github.com/huggingface/peft) by HuggingFace
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) by Tim Dettmers
