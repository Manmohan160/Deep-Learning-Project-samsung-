import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

# Load cleaned data
train_df = pd.read_csv("clean_train.csv")
dev_df = pd.read_csv("clean_dev.csv")
test_df = pd.read_csv("clean_test.csv")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# HuggingFace Dataset format
train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["ko"], text_target=examples["en"], padding="max_length", truncation=True, max_length=128)

# Apply tokenization
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_dev = dev_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Save tokenized datasets (optional)
tokenized_train.save_to_disk("tokenized_train")
tokenized_dev.save_to_disk("tokenized_dev")
tokenized_test.save_to_disk("tokenized_test")

print("✅ Tokenization complete. Datasets saved.")
# This script tokenizes the cleaned datasets using a multilingual BERT tokenizer and saves the tokenized datasets.