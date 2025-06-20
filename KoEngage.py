# KoEngage.py
# KoEngage.py

import torch
import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from tqdm import tqdm

# ✅ Load CSV
df = pd.read_csv("conversation_titles.csv")

# ✅ Load mBART model & tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Set language tokens
tokenizer.src_lang = "ko_KR"
target_lang_token = "en_XX"

# ✅ Translate each Korean sentence
translated_sentences = []
print("🔄 Translating...")
for sentence in tqdm(df["kor_title"].fillna("").astype(str)):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True).to(device)
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_token]
    )
    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    translated_sentences.append(translated)

# ✅ Save to DataFrame
df["translated_en"] = translated_sentences

# ✅ Save new file
df.to_csv("conversation_titles_with_translation.csv", index=False)
print("\n✅ Saved: conversation_titles_with_translation.csv")
