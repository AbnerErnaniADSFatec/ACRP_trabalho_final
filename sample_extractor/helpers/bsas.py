import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Menor/maior distância interna aos dados
def find_minmax_tau(x):
    m = x.shape[0]
    minTau, maxTau = np.inf, -1 * np.inf
    for i in range(m - 1):
        for j in range(i + 1, m):
            dist = np.linalg.norm(x[i,:] - x[j,:])
            if dist < minTau: minTau = dist
            if dist > maxTau: maxTau = dist
    return minTau, maxTau

def plot_centroids(X, labels, centroids, title = "Centroids"):
    labels = labels.astype(int)
    plt.figure(figsize=(8,8))
    # quantidade de clusters
    unique_labels = np.unique(labels)
    # plota cada cluster
    for k in unique_labels:
        cluster_points = X[labels == k]
        plt.scatter(
            cluster_points[:,0], cluster_points[:,1],
            label=f'Cluster {k}', alpha=0.7
        )
    # plota centróides
    plt.scatter(
        centroids[:,0], centroids[:,1],
        s=150,
        marker='o', c='red',
        label='Centroids'
    )
    plt.xlabel("$x_1$")
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Implementação do modelo BSAS
def BSAS(x, tau, maxClusters):
    c = 0  # Inicialização do contador de agrupamentos
    G = {} # Incialização de um dicionário
    ind = np.zeros(x.shape[0])-1 # Criação da lista de indicador de agrupamentos

    G[c] = []; G[c].append(x[0,:]) # Inclusão do primeiro exemplo
    vecMu = []; vecMu.append(x[0,:]) # Incialização do representante do G[1]
    ind[0] = c

    for i in range(1,x.shape[0]):
        diss = np.zeros(c + 1) # adição para compatibilizar 'c' com quantidade de agrupamentos
        for j in range(c + 1):
            diss[j] = np.linalg.norm(x[i,:] - vecMu[j])
        k = np.argmin(diss)

        if (diss[k] > tau) and (c < maxClusters-1):
            c += 1
            G[c] = []; G[c].append(x[i,:])
            ind[i] = c
            vecMu.append(x[i,:])
        else:
            G[k].append(x[i,:])
            ind[i] = k
            vecMu[k] = ((len(G[k]) -1)*vecMu[k] + x[i,:]) / len(G[k])

    # recalcula centróides finais corretamente
    centroids = []
    for k in np.unique(ind):
        points = x[ind == k]
        centroid = points.mean(axis = 0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    return ind, centroids

def search_clusters_bsas(X, steps = 100, repeticoes = 10):
    # Busca automatica de agrupamentos
    minTau, maxTau = find_minmax_tau(X)
    # steps = 200 # Número de avaliações no intervalo
    # repeticoes = 10 # Número de execuções para cada tau (devido à aleatoriedade)

    vecTau = np.linspace(minTau, maxTau, steps)
    vecAgrups = []
    for tau in vecTau:
        vec = []
        for _ in range(repeticoes):
            rand = np.argsort(np.random.randint(0,1,X.shape[0]))
            randX = np.copy(X[rand,:])
            res, _ = BSAS(X,tau,randX.shape[0])
            vec.append(np.unique(res).shape[0])
        vecAgrups.append(np.median(vec))

    return vecTau, vecAgrups

def elbow(vecTau, vecAgrups, eps = 1):
    # Suavização da curva
    smooth = gaussian_filter1d(vecAgrups, sigma=2)
    d1 = np.gradient(smooth, vecTau)
    d2 = np.gradient(d1, vecTau)

    # # limiar de estabilização
    # eps = 1

    # regiões estáveis
    stable = np.where(np.abs(d1) < eps)[0]

    # primeiro ponto estável
    idx_best = stable[0]

    best_tau = vecTau[idx_best]
    best_k = vecAgrups[idx_best]

    return {
        'best_tau': best_tau,
        'best_k': best_k,
        'd1': d1,
        'd2': d2,
        'stable_indices': stable,
        'idx_best': idx_best,
        'vecTau': vecTau,
        'vecAgrups': vecAgrups,
        'smooth': smooth
    }

def plot_elbow(elbow):
    best_tau = elbow['best_tau']
    best_k = elbow['best_k']
    vecTau = elbow['vecTau']
    vecAgrups = elbow['vecAgrups']
    smooth = elbow['smooth']

    plt.figure(figsize=(8,5))
    # curva original
    plt.plot(vecTau, vecAgrups, 'r-', alpha=0.4, label='Original')
    # curva suavizada
    plt.plot(vecTau, smooth, 'b-', linewidth=2, label='Suavizada')
    # cotovelo
    plt.scatter(best_tau, best_k, s=250, c='black', zorder=5, label='Cotovelo')
    # linha vertical
    plt.axvline(best_tau, linestyle='--', color='black')
    dx = (max(vecTau) - min(vecTau)) * 0.05
    dy = (max(vecAgrups) - min(vecAgrups)) * 0.05
    # anotação
    plt.annotate(
        rf'$\tau={best_tau:.2f}$' '\n'
        rf'$K={best_k:.0f}$',
        xy=(best_tau, best_k),
        xytext=(best_tau + dx, best_k + dy),
        arrowprops=dict(
            arrowstyle='->',
            lw=2
        ),
        fontsize=12,
        bbox=dict(
            boxstyle='round',
            fc='white'
        )
    )
    plt.xlabel(r'$\tau$', fontsize=20)
    plt.ylabel('N agrupamentos')
    plt.title('Elbow Method - BSAS')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
