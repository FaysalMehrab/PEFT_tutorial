from transformers import AutoTokenizer
from datasets import Dataset

checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


samples = [
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is overfitting?"},
            {"role": "assistant","content": "Overfitting is when a model memorizes training data including its noise, and fails on new data. Fix it with dropout, regularization or more data."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is a learning rate?"},
            {"role": "assistant","content": "Learning rate controls how big each update step is during training. Too high causes instability, too low is very slow. Typical range: 1e-3 to 1e-5."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is gradient descent?"},
            {"role": "assistant","content": "Gradient descent minimizes the loss by computing gradients and nudging weights in the direction that reduces error. Repeated thousands of times until the model converges."},
        ]
    },
]



def format_sample(sample):
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize = False,
        add_generation_prompt = False
    )
    return {"text": text}



dataset = Dataset.from_list(samples)
dataset = dataset.map(format_sample)

split = dataset.train_test_split(test_size = 0.2, seed = 42)

print(dataset)
print("\nFormated First sample:")
print(dataset[0]["text"])

print('-'*20)

print(split)
print(f"\nTrain samples: {len(split['train'])}")
print(f"Test samples: {len(split['test'])}")