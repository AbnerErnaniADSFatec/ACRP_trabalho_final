import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score

# Silhouette Score (melhor método prático)
def silhouette(X, k_range = (2, 20)):
    # matriz de linkage
    Z = linkage(X, method='ward')

    k_range = range(*k_range)
    scores = []

    for k in k_range:
        labels = fcluster(Z, k, criterion='maxclust')
        score = silhouette_score(X, labels)
        scores.append(score)

    # melhor k
    best_k = k_range[np.argmax(scores)]
    best_score = max(scores)

    # média
    mean_score = np.mean(scores)

    # índice do ponto mais próximo da média
    idx_closest = np.argmin(np.abs(np.array(scores) - mean_score))

    closest_k = k_range[idx_closest]
    closest_score = scores[idx_closest]

    return {
        'k_range': k_range,
        'scores': scores,
        'best_k': best_k,
        'best_score': best_score,
        'mean_score': mean_score,
        'closest_k': closest_k,
        'closest_score': closest_score
    }

def plot_silhouette(silhouette_results):

    k_range = silhouette_results['k_range']
    scores = silhouette_results['scores']
    best_k = silhouette_results['best_k']
    best_score = silhouette_results['best_score']
    mean_score = silhouette_results['mean_score']
    closest_k = silhouette_results['closest_k']
    closest_score = silhouette_results['closest_score']

    # plot
    plt.figure(figsize=(8,5))
    plt.plot(k_range, scores, marker='o')

    # melhor k
    plt.scatter(best_k, best_score, s=100)
    plt.axvline(best_k, linestyle='--', color = "red")

    # K próximo da média
    plt.scatter(best_k, best_score, s=100)
    plt.axvline(best_k, linestyle='--', color = "red")

    plt.scatter(closest_k, closest_score, s=100)
    plt.axvline(closest_k, linestyle='--')

    # média
    plt.axhline(mean_score, linestyle='--', color = "green")

    # anotação
    plt.text(best_k, best_score, f"k={best_k}", ha='left', va='bottom')
    plt.text(k_range.start, mean_score, f"mean={mean_score:.2f}", va='bottom')
    plt.text(closest_k, closest_score, f"k={closest_k}", ha='left', va='bottom')

    plt.title("Silhouette Score")
    plt.xlabel("Número de clusters")
    plt.ylabel("Score")

    plt.xticks(np.arange(min(k_range), max(k_range)+1, 2))
    plt.show()
