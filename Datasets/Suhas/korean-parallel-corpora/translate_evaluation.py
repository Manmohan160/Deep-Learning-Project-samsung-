# Save this as a Python file (e.g., translate_eval.py)

import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, T5ForConditionalGeneration, T5Tokenizer
import evaluate
import sys

# Load models and tokenizers (these will be loaded once when the script is run)
mbart_model_name = "facebook/mbart-large-50-many-to-many-mmt"
mbart_tokenizer = MBart50Tokenizer.from_pretrained(mbart_model_name)
mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_model_name)

t5_model_name = "google/flan/t5-small"  # You can change this to 't5-base', 't5-large', etc.
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# Load evaluation metrics (these will be loaded once when the script is run)
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
chrf = evaluate.load("chrf")


def generate_mbart_translations(texts, src_lang, tgt_lang):
    """Generates translations using mBart."""
    mbart_tokenizer.src_lang = src_lang
    encoded_input = mbart_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    generated_tokens = mbart_model.generate(**encoded_input, forced_bos_token_id=mbart_tokenizer.lang_code_to_id[tgt_lang])
    return mbart_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

def generate_t5_translations(texts, prefix):
    """Generates translations using T5."""
    # T5 often requires a prefix to specify the task
    prefixed_texts = [prefix + text for text in texts]
    encoded_input = t5_tokenizer(prefixed_texts, return_tensors="pt", padding=True, truncation=True)
    generated_tokens = t5_model.generate(**encoded_input)
    return t5_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def process_translation_file(file_path):
    """
    Loads a CSV file, performs translations and evaluations, and prints the results.

    Args:
        file_path (str): The path to the CSV file with 'en_text' and 'ko_text' columns.
    """
    print(f"\n--- Processing file: {file_path} ---")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # Get the texts for translation
    english_texts = df['en_text'].tolist()
    korean_texts = df['ko_text'].tolist()

    # Generate English to Korean translations
    mbart_translations_en_ko = generate_mbart_translations(english_texts, src_lang="en_XX", tgt_lang="ko_KR")
    t5_translations_en_ko = generate_t5_translations(english_texts, prefix="translate English to Korean: ")

    # Generate Korean to English translations
    mbart_translations_ko_en = generate_mbart_translations(korean_texts, src_lang="ko_KR", tgt_lang="en_XX")
    t5_translations_ko_en = generate_t5_translations(korean_texts, prefix="translate Korean to English: ")

    # Add translations to the DataFrame (optional, but good for inspection)
    df['mBart_translation_en_ko'] = mbart_translations_en_ko
    df['T5_translation_en_ko'] = t5_translations_en_ko
    df['mBart_translation_ko_en'] = mbart_translations_ko_en
    df['T5_translation_ko_en'] = t5_translations_ko_en

    # --- Evaluate English to Korean Translations ---
    references_ko_for_en_ko = df['ko_text'].tolist()
    mbart_translations_en_ko = df['mBart_translation_en_ko'].tolist()
    t5_translations_en_ko = df['T5_translation_en_ko'].tolist()

    print("\n--- Evaluating English to Korean Translations ---")

    mbart_bleu_results_en_ko = bleu.compute(predictions=mbart_translations_en_ko, references=references_ko_for_en_ko)
    t5_bleu_results_en_ko = bleu.compute(predictions=t5_translations_en_ko, references=references_ko_for_en_ko)
    print("mBart BLEU (en->ko):", mbart_bleu_results_en_ko)
    print("T5 BLEU (en->ko):", t5_bleu_results_en_ko)

    mbart_bertscore_results_en_ko = bertscore.compute(predictions=mbart_translations_en_ko, references=references_ko_for_en_ko, lang="ko")
    t5_bertscore_results_en_ko = bertscore.compute(predictions=t5_translations_en_ko, references=references_ko_for_en_ko, lang="ko")
    print("mBart BERTScore (en->ko):", mbart_bertscore_results_en_ko)
    print("T5 BERTScore (en->ko):", t5_bertscore_results_en_ko)

    mbart_chrf_results_en_ko = chrf.compute(predictions=mbart_translations_en_ko, references=references_ko_for_en_ko)
    t5_chrf_results_en_ko = chrf.compute(predictions=t5_translations_en_ko, references=references_ko_for_en_ko)
    print("mBart chrF++ (en->ko):", mbart_chrf_results_en_ko)
    print("T5 chrF++ (en->ko):", t5_chrf_results_en_ko)

    # --- Evaluate Korean to English Translations ---
    references_en_for_ko_en = df['en_text'].tolist()
    mbart_translations_ko_en = df['mBart_translation_ko_en'].tolist()
    t5_translations_ko_en = df['T5_translation_ko_en'].tolist()

    print("\n--- Evaluating Korean to English Translations ---")

    mbart_bleu_results_ko_en = bleu.compute(predictions=mbart_translations_ko_en, references=references_en_for_ko_en)
    t5_bleu_results_ko_en = bleu.compute(predictions=t5_translations_ko_en, references=references_en_for_ko_en)
    print("mBart BLEU (ko->en):", mbart_bleu_results_ko_en)
    print("T5 BLEU (ko->en):", t5_bleu_results_ko_en)

    mbart_bertscore_results_ko_en = bertscore.compute(predictions=mbart_translations_ko_en, references=references_en_for_ko_en, lang="en")
    t5_bertscore_results_ko_en = bertscore.compute(predictions=t5_translations_ko_en, references=references_en_for_ko_en, lang="en")
    print("mBart BERTScore (ko->en):", mbart_bertscore_results_ko_en)
    print("T5 BERTScore (ko->en):", t5_bertscore_results_ko_en)

    mbart_chrf_results_ko_en = chrf.compute(predictions=mbart_translations_ko_en, references=references_en_for_ko_en)
    t5_chrf_results_ko_en = chrf.compute(predictions=t5_translations_ko_en, references=references_en_for_ko_en)
    print("mBart chrF++ (ko->en):", mbart_chrf_results_ko_en)
    print("T5 chrF++ (ko->en):", t5_chrf_results_ko_en)

if __name__ == "__main__":
    file_list = sys.argv[1:]

    if not file_list:
        print("Usage: python your_script_name.py file1.csv file2.csv ...")
        sys.exit(1)

    for file in file_list:
        process_translation_file(file)