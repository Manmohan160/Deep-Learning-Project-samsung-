import pandas as pd
import argparse

def create_bilingual_dataset_from_separate_files(en_file_path, ko_file_path, output_filename="bible_dataset.csv"):
   
    # Read English texts
    with open(en_file_path, 'r', encoding='utf-8') as f:
        en_lines = f.readlines()
    
    # Read Korean texts
    with open(ko_file_path, 'r', encoding='utf-8') as f:
        ko_lines = f.readlines()

    # Clean and store texts
    en_texts_cleaned = []
    ko_texts_cleaned = []

    # Process English lines
    for line in en_lines:
        cleaned_line = line.strip()
        # Remove the prefix (e.g., "Genesis1.1 ") if it exists
        # if ' ' in cleaned_line and not cleaned_line.startswith(' '): # Ensure there's a space and it's not just a space at the beginning
        #     parts = cleaned_line.split(' ', 1)
        #     en_texts_cleaned.append(parts[1].strip())
        # else:
        #   en_texts_cleaned.append(cleaned_line) # If no prefix or just spaces, keep as is
        en_texts_cleaned.append(cleaned_line)

    # Process Korean lines
    for line in ko_lines:
        cleaned_line = line.strip()
        # Remove the prefix (e.g., "Genesis1.1 ") if it exists
        # if ' ' in cleaned_line and not cleaned_line.startswith(' '):
        #     parts = cleaned_line.split(' ', 1)
        #     ko_texts_cleaned.append(parts[1].strip())
        # else:
        #     ko_texts_cleaned.append(cleaned_line)
        ko_texts_cleaned.append(cleaned_line)

    # Ensure both lists have the same number of elements
    min_len = min(len(en_texts_cleaned), len(ko_texts_cleaned))
    en_texts_final = en_texts_cleaned[:min_len]
    ko_texts_final = ko_texts_cleaned[:min_len]

    df = pd.DataFrame({
        'en_text': en_texts_final,
        'ko_text': ko_texts_final
    })

    df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"Dataset saved to {output_filename}")
    print(df.head()) # Display the first few rows of the DataFrame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a bilingual dataset from separate English and Korean text files.")
    parser.add_argument('en_file', type=str, help='Path to the English text file.')
    parser.add_argument('ko_file', type=str, help='Path to the Korean text file.')
    parser.add_argument('-o', '--output', type=str, default='bilingual_dataset.csv',
                        help='Name of the output CSV file (default: bilingual_dataset.csv).')

    args = parser.parse_args()

    create_bilingual_dataset_from_separate_files(args.en_file, args.ko_file, args.output)