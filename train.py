from datasets import load_dataset, Dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import pandas as pd
import torch

# ✅ Load and convert CSV to HuggingFace Dataset
df = pd.read_csv("koengage_dataset.csv").dropna()
hf_dataset = Dataset.from_pandas(df)

# ✅ Load model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# ✅ Set source and target language codes
# ✅ Set language codes for mBART
tokenizer.src_lang = "ko_KR"
tokenizer.tgt_lang = "en_XX"  # ✅ This line fixes the KeyError
target_lang = "en_XX"


# ✅ Preprocessing
def preprocess(example):
    inputs = tokenizer(example["kor_title"], padding="max_length", truncation=True, max_length=64)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["eng_title"], padding="max_length", truncation=True, max_length=64)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = hf_dataset.map(preprocess, batched=False)

# ✅ Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/mbart-koengage",
    evaluation_strategy="no",
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=5,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),  # use FP16 if GPU available
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
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ✅ Train
trainer.train()
