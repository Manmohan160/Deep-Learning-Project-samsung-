from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import pandas as pd
import torch
from tqdm import tqdm  # ✅ progress bar

# ✅ Load fine-tuned model and tokenizer
model_path = "./mbart-koengage-finetuned"
model = MBartForConditionalGeneration.from_pretrained(model_path)
tokenizer = MBart50TokenizerFast.from_pretrained(model_path)

# ✅ Set language codes
tokenizer.src_lang = "ko_KR"
forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]

# ✅ Load original data
df = pd.read_csv("conversation_titles.csv")
df = df[['kor_title', 'eng_title']].dropna()
df.columns = ['ko', 'en']

# ✅ Translate with progress bar
translated = []
for text in tqdm(df["ko"], desc="Translating", ncols=100):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translated.append(translated_text)

# ✅ Save output
df["pred_en_finetuned"] = translated
df.to_csv("conversation_titles_with_finetuned_translation.csv", index=False)
print("✅ Translations saved to conversation_titles_with_finetuned_translation.csv")
