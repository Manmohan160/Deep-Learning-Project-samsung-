import pandas as pd
from evaluate import load

# Load the generated translations
df = pd.read_csv("translation_output/generated_translations.csv")

# Remove rows with missing values
df.dropna(subset=["en", "generated_en"], inplace=True)

# Prepare references and predictions
references = df["en"].tolist()
predictions = df["generated_en"].tolist()

# Load metrics
bleu = load("sacrebleu", module_type="metric")
bertscore = load("bertscore", module_type="metric")
chrf = load("chrf", module_type="metric")

# Compute BLEU
try:
    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    print(f"\n🔵 BLEU score: {bleu_score['bleu']:.4f}")
except Exception as e:
    print(f"❌ BLEU computation failed: {e}")

# Compute BERTScore
try:
    bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
    avg_bertscore = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])
    print(f"🟢 BERTScore (F1): {avg_bertscore:.4f}")
except Exception as e:
    print(f"❌ BERTScore computation failed: {e}")

# Compute chrF++
try:
    chrf_result = chrf.compute(predictions=predictions, references=references)
    print(f"🟡 chrF++ score: {chrf_result['score']:.4f}")
except Exception as e:
    print(f"❌ chrF++ computation failed: {e}")
