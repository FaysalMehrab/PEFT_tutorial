# PEFT / QLoRA Fine-tuning Study — SmolLM2 135M

Learning Parameter Efficient Fine-Tuning (PEFT) from scratch using LoRA and QLoRA on SmolLM2-135M-Instruct.

---

## What's Inside

| File/Notebook | Description | Environment |
|---|---|---|
| `tokenizer_test.py` | Tokenization, subword splitting, ChatML format | Local CPU |
| `model_test.py` | Model architecture, parameter count, LoRA attachment | Local CPU |
| `dataset_test.py` | Dataset preparation, chat template formatting | Local CPU |
| `train.py` | Full LoRA training pipeline (CPU) | Local CPU |
| `inference.py` | Load adapter + run inference | Local CPU |
| `qlora_kaggle.ipynb` | Full QLoRA pipeline with Unsloth on GPU | Kaggle T4 |

---

## Key Results

| | Local (CPU) | Kaggle (GPU) |
|---|---|---|
| Method | LoRA (no quantization) | True QLoRA (4-bit) |
| Trainable params | 1.8M / 134.5M (1.35%) | 3.2M / 137.7M (2.38%) |
| Adapter size | 7.1 MB vs 513 MB full model | 7.1 MB |
| Training loss | N/A (too small dataset) | ~1.68 → 1.73 |
| Framework | transformers + peft + trl | Unsloth + trl |

---

## Local Setup (CPU)

```bash
git clone https://github.com/FaysalMehrab/PEFT_tutorial.git
cd PEFT_tutorial

python3 -m venv peft_env
source peft_env/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft trl datasets accelerate
```

### Run locally
```bash
python3 tokenizer_test.py   # understand tokenization
python3 model_test.py       # explore model + attach LoRA
python3 dataset_test.py     # prepare dataset
python3 train.py            # train (CPU, ~20 mins)
python3 inference.py        # chat with fine-tuned model
```

> ⚠️ CPU-only = plain LoRA, not QLoRA. QLoRA requires a CUDA GPU.

---

## Kaggle Notebook (GPU — True QLoRA)

### Setup
1. Go to [kaggle.com/code](https://kaggle.com/code) → New Notebook
2. Settings → Accelerator → **GPU T4 x2**
3. Settings → Internet → **On**
4. Open `qlora_kaggle.ipynb`

### Install
```python
!pip install unsloth
```

### Pipeline Overview
```
Load model (4-bit QLoRA)
    ↓
Attach LoRA adapters (r=16, target all attention + MLP layers)
    ↓
Load dataset (HuggingFaceH4/ultrachat_200k, 300 samples)
    ↓
Format with ChatML template
    ↓
Train with SFTTrainer (3 epochs)
    ↓
Save adapter → /kaggle/working/output/final
Save merged → /kaggle/working/output/merged
    ↓
Inference with FastLanguageModel
```

### Key Hyperparameters
```python
# LoRA
r = 16
lora_alpha = 32
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training
learning_rate = 2e-4
num_train_epochs = 3
per_device_train_batch_size = 2
gradient_accumulation_steps = 4   # effective batch = 8
optimizer = "adamw_8bit"
```

---

## Concepts Learned

**Tokenization**
- Subword tokenization — rare/technical words split into pieces
- ChatML format — how system/user/assistant turns are structured
- `add_generation_prompt=True` for inference, `False` for training

**Model Architecture**
- Embeddings → 30 × LlamaDecoderLayer → lm_head
- Each layer: self-attention (q/k/v/o) + MLP (gate/up/down)
- 134.5M total parameters, 513MB in float32

**LoRA Math**
```
Instead of updating W (576×576 = 331,776 params)
LoRA adds:  A (576×16) + B (16×576) = 18,432 params
Saving: 94% fewer parameters per layer
```

**QLoRA vs LoRA**
```
LoRA  = Frozen base model (float16) + trainable adapters
QLoRA = Frozen base model (4-bit)   + trainable adapters (float16)
```

**Two Save Methods**
```
adapter only  →  7MB,   needs base model at inference
merged model  →  270MB, self-contained, ready to deploy
```

---

## Requirements

- Python 3.10+
- Local: No GPU needed (CPU only)
- Kaggle: T4 GPU (free), internet enabled

---

## Model

[HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) — 135M parameter instruction-tuned model, small enough to run on CPU for learning purposes.
