import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from esm.inverse_folding import plot_contact_map

# Load chirality evaluation results
chirality_scores = pd.read_csv("results/chirality_scores.csv")

# Load RoseTTAFold model predictions
predicted_scores = ...  # Extract predicted scores from the RoseTTAFold model

# Calculate R^2 and Spearman correlation
r2 = r2_score(chirality_scores["true_score"], predicted_scores)
spearman_corr, _ = spearmanr(chirality_scores["true_score"], predicted_scores)

print(f"R^2: {r2:.3f}")
print(f"Spearman correlation: {spearman_corr:.3f}")

# Plot true vs. predicted scores
plt.figure(figsize=(6, 6))
plt.scatter(chirality_scores["true_score"], predicted_scores, alpha=0.6)
plt.xlabel("True chirality score")
plt.ylabel("Predicted score")
plt.title("RoseTTAFold model performance")
plt.savefig("results/model_performance.png", dpi=300)
plt.show()

# Load encoder output and attention maps
encoder_output = torch.load("results/encoder_output.pt")
attention_map = torch.load("results/attention_map.pt")

# Perform PCA on encoder output
pca = PCA(n_components=2)
encoder_output_pca = pca.fit_transform(encoder_output)

plt.figure(figsize=(6, 6))
plt.scatter(encoder_output_pca[:, 0], encoder_output_pca[:, 1], alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of RoseTTAFold encoder output")
plt.savefig("results/encoder_output_pca.png", dpi=300)
plt.show()

# Visualize attention maps
plot_contact_map(attention_map, "results/attention_map.png")