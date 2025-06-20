# ✅ evaluate_testset_metrics.py — Evaluates BLEU and BERTScore for test set

import pandas as pd
from evaluate import load

# ✅ Load CSV with test predictions
df = pd.read_csv("conversation_titles_with_testset_predictions.csv").dropna()

references = df["eng_title"].tolist()            # Ground truth
predictions = df["pred_en_testset"].tolist()     # Model output

# ✅ Evaluate BLEU
print("\U0001F4D8 Computing BLEU Score...")
bleu = load("sacrebleu")
bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
print(f"\n📘 BLEU Score: {bleu_result['score']:.2f}\n")

# ✅ Evaluate BERTScore
print("\U0001F52C Computing BERTScore...")
bertscore = load("bertscore")
bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
print(f"🔬 BERTScore → Precision: {sum(bertscore_result['precision']) / len(bertscore_result['precision']):.4f}, ",
      f"Recall: {sum(bertscore_result['recall']) / len(bertscore_result['recall']):.4f}, ",
      f"F1: {sum(bertscore_result['f1']) / len(bertscore_result['f1']):.4f}")
