from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch

checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    torch_dtype = torch.float32,
    device_map = 'cpu'
)

# print(model)


# total = sum(p.numel() for p in model.parameters())

# trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Total: {total:,}")
# print(f"Trainable: {trainable:,}")
# print(f"Model Size (float32): {total * 4 / 1024**2:.2f} MB")

lora_config = LoraConfig(
    r = 16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"Total: {total:,}")
print(f"Trainable: {trainable:,}")
print(f"Trainable%: {100 * trainable / total:.2f} %")

model.print_trainable_parameters()