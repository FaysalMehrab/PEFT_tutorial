from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"


tokenizer = AutoTokenizer.from_pretrained(checkpoint)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    torch_dtype = torch.float32,
    device_map = 'cpu'

)

lora_config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


samples = [
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is a neural network?"},
            {"role": "assistant","content": "A neural network is a set of interconnected layers of nodes (neurons) that transform input data into output predictions through learned weights and activations."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is a bias in ML?"},
            {"role": "assistant","content": "Bias is an extra parameter in neurons that allows the model to shift activations, enabling it to fit data better."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is a feature in machine learning?"},
            {"role": "assistant","content": "A feature is an individual measurable property or characteristic of the data used as input to a model."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is cross-validation?"},
            {"role": "assistant","content": "Cross-validation splits data into multiple folds to train and validate a model on different subsets, improving reliability of performance estimates."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is a decision tree?"},
            {"role": "assistant","content": "A decision tree predicts by splitting data recursively based on features, creating a tree structure where leaves are output predictions."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is ensemble learning?"},
            {"role": "assistant","content": "Ensemble learning combines multiple models to improve accuracy and robustness, e.g., bagging, boosting, or stacking."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is a support vector machine (SVM)?"},
            {"role": "assistant","content": "SVM is a model that finds the hyperplane maximizing the margin between classes in a feature space, effective for classification."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is a confusion matrix?"},
            {"role": "assistant","content": "A confusion matrix shows counts of true positives, true negatives, false positives, and false negatives, summarizing classification performance."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is precision and recall?"},
            {"role": "assistant","content": "Precision measures correct positive predictions out of all predicted positives. Recall measures correct positives out of all actual positives."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is a ROC curve?"},
            {"role": "assistant","content": "A ROC curve plots true positive rate vs false positive rate at different thresholds, helping evaluate classifier performance."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is a ReLU activation?"},
            {"role": "assistant","content": "ReLU outputs the input if positive, otherwise zero. It speeds up training and reduces vanishing gradient problems."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is softmax?"},
            {"role": "assistant","content": "Softmax converts raw scores into probabilities that sum to 1, commonly used in the output layer for multi-class classification."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is dropout?"},
            {"role": "assistant","content": "Dropout randomly disables neurons during training to prevent overfitting and improve generalization."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is overfitting vs underfitting?"},
            {"role": "assistant","content": "Overfitting memorizes training data and fails on new data; underfitting cannot capture patterns and performs poorly even on training data."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is a learning curve?"},
            {"role": "assistant","content": "A learning curve plots training and validation performance over epochs, showing how well a model is learning."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is a hidden layer?"},
            {"role": "assistant","content": "A hidden layer is an intermediate layer in a neural network between input and output, where computations extract features."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is a loss function?"},
            {"role": "assistant","content": "A loss function measures the difference between predicted and true values; minimizing it guides model training."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is stochastic gradient descent (SGD)?"},
            {"role": "assistant","content": "SGD updates model weights using gradients from a random subset (mini-batch) of data, making training faster and more stochastic."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is a parameter vs hyperparameter?"},
            {"role": "assistant","content": "Parameters are learned by the model (e.g., weights), while hyperparameters are set manually (e.g., learning rate, number of layers)."},
        ]
    },
    {
        "messages": [
            {"role": "system",  "content": "You are a concise ML tutor."},
            {"role": "user",    "content": "What is data augmentation?"},
            {"role": "assistant","content": "Data augmentation artificially increases dataset size by applying transformations like rotation, scaling, or flipping to improve generalization."},
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
dataset = dataset.map(format_sample, remove_columns=["messages"])

split = dataset.train_test_split(test_size = 0.2, seed = 42)

# print(split)
# print(f"\nTrain samples: {len(split['train'])}")
# print(f"Test samples:  {len(split['test'])}")
# print(f"\nSample training text:\n{split['train'][0]['text']}")


training_args = SFTConfig(
    output_dir = "./output",
    num_train_epochs = 3,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps= 4,
    learning_rate = 2e-4,
    max_length = 512,
    logging_steps = 5,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    dataset_text_field = "text",
    fp16 = False,
    use_cpu = True,
    optim = "adamw_torch",
    report_to = "none",
)

trainer = SFTTrainer(
    model = model,
    train_dataset = split["train"],
    eval_dataset = split["test"],
    args = training_args,
    processing_class = tokenizer,
)




trainer.model.save_pretrained("./output/final")
tokenizer.save_pretrained("./output/final")
print("Model saved!")