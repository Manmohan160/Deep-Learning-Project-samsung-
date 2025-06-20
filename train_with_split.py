# ✅ Updated train.py with Train/Test Split
from datasets import Dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# ✅ Load and split dataset
df = pd.read_csv("koengage_dataset.csv").dropna()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save test set for evaluation later
test_df.to_csv("koengage_testset.csv", index=False)

# ✅ Convert train set to HuggingFace Dataset
hf_train_dataset = Dataset.from_pandas(train_df)

# ✅ Load model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# ✅ Set language codes for mBART
tokenizer.src_lang = "ko_KR"
tokenizer.tgt_lang = "en_XX"

# ✅ Preprocessing function
def preprocess(example):
    inputs = tokenizer(example["kor_title"], padding="max_length", truncation=True, max_length=64)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["eng_title"], padding="max_length", truncation=True, max_length=64)
    inputs["labels"] = labels["input_ids"]
    return inputs

# ✅ Tokenize training data
tokenized_train_dataset = hf_train_dataset.map(preprocess, batched=False)

# ✅ Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/mbart-koengage",
    evaluation_strategy="no",
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=5,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    save_strategy="epoch",
    report_to="none"
)

# ✅ Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ✅ Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ✅ Train
trainer.train()

# ⚠️ Don't forget to use "koengage_testset.csv" in your inference script next
