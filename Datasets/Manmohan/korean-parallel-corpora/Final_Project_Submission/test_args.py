from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    output_dir="./test_output",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    logging_dir="./logs"
)

print("✅ Passed: Seq2SeqTrainingArguments accepted all expected arguments")

# Check if the arguments are set correctly