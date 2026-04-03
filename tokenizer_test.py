from transformers import AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text = "What is LoRA?"

tokens = tokenizer(text)

print(f'Text: {text}')
print(f'Token IDs: {tokens["input_ids"]}')
print(f'Decode Back: {tokenizer.decode(tokens["input_ids"])}')


for id in tokens["input_ids"]:
    print(f"{id} --> '{tokenizer.decode([id])}'")


print(f"Vocab Size: {tokenizer.vocab_size}")

print('-'*10)

bangla = "আমি মেশিন লার্নিং শিখছি"
bangla_tokens = tokenizer(bangla)

print(f"Text: {bangla}")
for id_1 in bangla_tokens["input_ids"]:
    print(f"{id_1} --> '{tokenizer.decode([id_1])}'")

print(f"Decode Back: {tokenizer.decode(bangla_tokens['input_ids'])}")
print(f"Vocab Size: {tokenizer.vocab_size}")