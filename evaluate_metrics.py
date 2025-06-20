# evaluate_metrics.py

import pandas as pd
from sacrebleu import corpus_bleu
from bert_score import score

# ✅ Load translated file
df = pd.read_csv("conversation_titles_with_translation.csv")

# ✅ Extract references and predictions
references = df["eng_title"].fillna("").tolist()        # Ground truth
predictions = df["translated_en"].fillna("").tolist()   # mBART output

# ✅ BLEU Evaluation
bleu = corpus_bleu(predictions, [references])
print(f"📘 BLEU Score: {bleu.score:.2f}")

# ✅ BERTScore Evaluation
P, R, F1 = score(predictions, references, lang="en", verbose=True)
print(f"🔬 BERTScore → Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
