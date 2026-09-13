# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula04-v1.pptx
# Slide 56: VISUALIZAÇÃO EM CÓDIGO CLUSTERS SEMÂNTICOS COM T-SNE
# Bloco 10

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# Selecionamos um conjunto de palavras de interesse
palavras = ['cachorro', 'gato', 'leão', 'tigre', 'elefante',
            'carro', 'moto', 'ônibus', 'caminhão', 'bicicleta',
            'feliz', 'triste', 'alegre', 'deprimido', 'eufórico']
# Cada palavra tem vetor de 300 dimensões
vetores = np.array([modelo[p] for p in palavras])
print(vetores.shape)  # (15, 300)
# Reduz para 2D com t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
vetores_2d = tsne.fit_transform(vetores)
print(vetores_2d.shape)  # (15, 2)
# Plota
plt.figure(figsize=(10, 8))
plt.scatter(vetores_2d[:, 0], vetores_2d[:, 1])
for i, palavra in enumerate(palavras):
    plt.annotate(palavra, (vetores_2d[i, 0], vetores_2d[i, 1]))
plt.title('Clusters semânticos em PT-BR (FastText)')
plt.savefig('tsne_clusters.png', dpi=150)
