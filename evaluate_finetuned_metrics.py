import pandas as pd
from sacrebleu import corpus_bleu
from bert_score import score
from tqdm import tqdm

# ✅ Load CSV
df = pd.read_csv("conversation_titles_with_finetuned_translation.csv")

# ✅ Extract references and predictions
references = df["en"].fillna("").tolist()                 # Ground truth
predictions = df["pred_en_finetuned"].fillna("").tolist() # Fine-tuned output

# ✅ Progress bar (just visual)
print("📘 Computing BLEU Score...")
for _ in tqdm(range(len(predictions)), desc="→ BLEU Eval"):
    pass
bleu = corpus_bleu(predictions, [references])
print(f"\n📘 BLEU Score: {bleu.score:.2f}")

# ✅ BERTScore
print("🔬 Computing BERTScore...")
P, R, F1 = score(predictions, references, lang="en", verbose=True)
print(f"🔬 BERTScore → Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
