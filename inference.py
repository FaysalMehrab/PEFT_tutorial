from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


base_model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    torch_dtype = torch.float32,
    device_map = 'cpu'
)
model = PeftModel.from_pretrained(base_model, "./output/final")
model.eval()


def generate_response(user_message):
    messages = [
        {"role": "system", "content": "You are a concise ML tutor."},
        {"role": "user",   "content": user_message},
    ]
        # 1. Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_promt = True
    )
    
    # 2. Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 3. Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens = 500,
            temperature = 0.3,
            do_sample = True,
            pad_token_id = tokenizer.eos_token_id,
        )
    
    # 4. Decode & return
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response
print(generate_response("What is dropout in ML?"))