import matplotlib.pyplot as plt

# Replace with your actual scores
bleu_score = 7.41      # replace if fixed
bertscore_f1 = 0.7481  # already computed
chrf_score = 7.4129    # already computed

# Metric names and values
metrics = ['BLEU', 'BERTScore (F1)', 'chrF++']
values = [bleu_score, bertscore_f1 * 100, chrf_score]  # scale BERTScore to %

# Plot setup
plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=['skyblue', 'mediumseagreen', 'gold'])

# Annotate bars with values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}", ha='center', fontsize=12)

# Title and axis labels
plt.title("Translation Quality Metrics", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.ylim(0, max(values) + 10)

# Save and show
plt.tight_layout()
plt.savefig("translation_output/translation_metrics_barplot.png", dpi=300)
plt.show()
