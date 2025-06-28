from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_from_disk
import evaluate

# Load tokenized datasets
train_dataset = load_from_disk("tokenized_train")
dev_dataset = load_from_disk("tokenized_dev")

# Load model & tokenizer
model_name = "Helsinki-NLP/opus-mt-ko-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Data collator for padding dynamically
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Metric for evaluation
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = [[tokenizer.decode(l, skip_special_tokens=True)] for l in labels]
    return metric.compute(predictions=decoded_preds, references=labels)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./translation_output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
)

# Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save model
model.save_pretrained("fine_tuned_marianmt_ko_en")
tokenizer.save_pretrained("fine_tuned_marianmt_ko_en")
print("✅ Fine-tuning complete. Model and tokenizer saved as 'fine_tuned_marianmt_ko_en'.")