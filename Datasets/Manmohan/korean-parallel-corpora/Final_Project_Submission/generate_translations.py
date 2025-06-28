# generate_translations.py

from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from tqdm import tqdm
import torch

# Load fine-tuned model and tokenizer
model_path = "./fine_tuned_marianmt_ko_en"
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load test dataset
df = pd.read_csv("clean_test.csv")
source_texts = df["ko"].tolist()

# Generate translations
translated_texts = []
batch_size = 8  # adjust based on your system capability

print("Generating translations...")
for i in tqdm(range(0, len(source_texts), batch_size)):
    batch = source_texts[i:i + batch_size]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        translated = model.generate(**inputs, max_length=512, num_beams=6)
    decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
    translated_texts.extend(decoded)

# Save the results
df["generated_en"] = translated_texts
df.to_csv("translation_output/generated_translations.csv", index=False)

print("✅ Translations saved to: translation_output/generated_translations.csv")
