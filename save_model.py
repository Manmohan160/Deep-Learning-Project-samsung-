from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# ✅ Path to your best checkpoint
checkpoint_path = "./models/mbart-koengage/checkpoint-1065"

# ✅ Load model and tokenizer
model = MBartForConditionalGeneration.from_pretrained(checkpoint_path)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# ✅ Save to a clean final directory
save_path = "./mbart-koengage-finetuned"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("✅ Fine-tuned model saved to:", save_path)
