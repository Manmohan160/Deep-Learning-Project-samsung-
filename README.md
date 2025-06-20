# Group 10- Deep-Learning-Project-samsung

This repository contains the implementation, datasets, and evaluation results for the **KoEngage** project — a bilingual machine translation system developed as part of the Deep Learning collaboration between **IISc** and **Samsung Research**.

The goal of KoEngage is to translate **Korean → English** conversation-style titles using both **prompt-based** and **fine-tuned** transformer models (mBART50, T5).

---

##  Repository Structure

| File/Folder                         | Purpose                                                                 |
|------------------------------------|-------------------------------------------------------------------------|
| `koengage_dataset.csv`             | Full bilingual dataset (Korean-English pairs)                           |
| `koengage_testset.csv`             | Held-out test set (20% split from dataset)                              |
| `conversation_titles_with_*.csv`   | Inference outputs for prompt-based, fine-tuned, and test set models     |
| `train.py`, `train_with_split.py`  | Scripts to fine-tune mBART model                                        |
| `evaluate_metrics.py`, etc.        | Scripts to compute BLEU, BERTScore, and chrF++                          |
| `predict.py`, `inference_*.py`     | Inference scripts for model output generation                           |
| `requirements.txt`                 | Required Python packages (offline use via .whl available)               |
| `.gitignore`                       | Git cleanup to exclude unnecessary virtual env, cache, and logs         |

---

##  Evaluation Summary

| Phase                   | # Pairs               | Words (en) | Model                           | Method                        | Data Leakage | BLEU  | BERTScore (F1) |
|-------------------------|------------------------|------------|----------------------------------|-------------------------------|---------------|-------|----------------|
| **Prompt-Only**         | 849                    | ~6,100     | mBART50 (Pretrained)             | Prompt-based inference        | ❌            | ~19.50| ~0.860         |
| **Fine-Tuned (No Split)**| 849                    | ~6,100     | mBART50 (Fine-tuned on full set) | Full fine-tune on all data    | ✅            | 97.81 | 0.9981         |
| **Fine-Tuned (Split)**  | 679 (train)<br>170 (test)| ~5,100    | mBART50 (Fine-tuned)             | Fine-tune on 679 + evaluate 170 | ❌         | 46.03 | 0.9579         |

---

##  Key Insights

- **Prompting** is fast but underperforms due to lack of task adaptation.
- **Full fine-tuning without a test split** gives inflated scores due to data leakage.
- **Split-based fine-tuning** provides a realistic metric on unseen data.
- **BERTScore** shows high semantic accuracy even when BLEU is moderate — useful in low-resource setups.

---

##  Setup Instructions

```bash
# Clone the repo
git clone https://github.com/Manmohan160/Deep-Learning-Project-samsung-.git

# (Optional) Create virtual env
python -m venv koengage_env
.\koengage_env\Scripts\activate

# Install requirements (offline or with pip)
pip install -r requirements.txt
```

---

##  License

This project is intended for academic use under the IISc-Samsung collaboration. Please contact the authors for any external use.

---

##  Acknowledgements

- IISc Department of Computational and Data Sciences
- Samsung Research Team
- Deepak Sir, Team 10, and KoEngage contributors

---
