from Korpora import Korpora
import pandas as pd

# Step 1: Download the corpus
Korpora.fetch("korean_parallel_koen_news")

# Step 2: Load the corpus
corpus = Korpora.load("korean_parallel_koen_news")

# Step 3: Extract datasets
train = corpus.train
dev = corpus.dev
test = corpus.test

# Step 4: Convert to DataFrames
train_df = pd.DataFrame({"ko": [x.text for x in train], "en": [x.pair for x in train]})
dev_df   = pd.DataFrame({"ko": [x.text for x in dev],   "en": [x.pair for x in dev]})
test_df  = pd.DataFrame({"ko": [x.text for x in test],  "en": [x.pair for x in test]})

# Step 5: Print a few samples
print("Sample from training data:")
print(train_df.sample(5))

# Optional: Save to CSV for later use
train_df.to_csv("train_koen.csv", index=False)
dev_df.to_csv("dev_koen.csv", index=False)
test_df.to_csv("test_koen.csv", index=False)
