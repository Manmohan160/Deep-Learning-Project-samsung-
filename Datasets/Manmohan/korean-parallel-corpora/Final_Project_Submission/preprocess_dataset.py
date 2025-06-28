import pandas as pd

# Load CSV files
train_df = pd.read_csv("train_koen.csv")
dev_df = pd.read_csv("dev_koen.csv")
test_df = pd.read_csv("test_koen.csv")

# Function to clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.strip()
    text = text.replace('\u3000', ' ')  # remove full-width space
    return text

# Clean and filter data
def preprocess(df, max_len=150):
    df['ko'] = df['ko'].astype(str).map(clean_text)
    df['en'] = df['en'].astype(str).map(clean_text)
    df = df[(df['ko'].str.len() > 0) & (df['en'].str.len() > 0)]
    df = df[(df['ko'].str.split().str.len() < max_len) & (df['en'].str.split().str.len() < max_len)]
    return df.reset_index(drop=True)

# Apply preprocessing
train_df = preprocess(train_df)
dev_df = preprocess(dev_df)
test_df = preprocess(test_df)

# Save cleaned datasets
train_df.to_csv("clean_train.csv", index=False)
dev_df.to_csv("clean_dev.csv", index=False)
test_df.to_csv("clean_test.csv", index=False)

print("✅ Preprocessing complete. Cleaned files saved as:")
print("  - clean_train.csv")
print("  - clean_dev.csv")
print("  - clean_test.csv")
# This script preprocesses the datasets by cleaning text and filtering based on length.