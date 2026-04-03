# LoRA Fine-tuning — SmolLM2 135M (Local CPU)

Fine-tuning a small LLM using LoRA (Parameter Efficient Fine-Tuning) on a local CPU machine.

## Model
`HuggingFaceTB/SmolLM2-135M-Instruct`

## Setup
```bash
python3 -m venv peft_env
source peft_env/bin/activate
pip install torch transformers peft trl datasets accelerate
```

## Files
| File | Purpose |
|---|---|
| `tokenizer_test.py` | Explore tokenization and ChatML format |
| `model_test.py` | Load model, count parameters, attach LoRA |
| `dataset_test.py` | Build dataset, format with chat template |
| `train.py` | Full LoRA training pipeline |
| `inference.py` | Load adapter and run inference |

## Run
```bash
python3 train.py    # Train
python3 inference.py  # Inference
```

## Results
- Trainable parameters reduced from **134.5M → 1.8M (1.35%)**
- LoRA adapter size: **7.1MB** vs 513MB full model
- Note: CPU-only = LoRA, not QLoRA (QLoRA requires GPU)

## Requirements
- Python 3.10+
- No GPU required