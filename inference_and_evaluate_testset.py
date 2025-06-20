from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import pandas as pd
import torch
from tqdm import tqdm

# ✅ Load test set
df_test = pd.read_csv("koengage_testset.csv").dropna()

# ✅ Load the fine-tuned model
model_path = "./models/mbart-koengage"
model = MBartForConditionalGeneration.from_pretrained(model_path)
tokenizer = MBart50Tokenizer.from_pretrained(model_path)

# ✅ Set source and target languages
tokenizer.src_lang = "ko_KR"
tokenizer.tgt_lang = "en_XX"

# ✅ Move model to available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

translated_texts = []

# ✅ Inference with batching
batch_size = 32
with torch.no_grad():
    for i in tqdm(range(0, len(df_test), batch_size), desc="Translating..."):
        batch_texts = df_test["kor_title"].tolist()[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)

        translated_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=200,
            num_beams=5,
            early_stopping=True,
            forced_eos_token_id=tokenizer.eos_token_id
        )

        batch_translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translated_texts.extend(batch_translations)

# ✅ Save to CSV
df_test["pred_en_testset"] = translated_texts
df_test.to_csv("conversation_titles_with_testset_predictions.csv", index=False)

print("✅ Saved to conversation_titles_with_testset_predictions.csv")
