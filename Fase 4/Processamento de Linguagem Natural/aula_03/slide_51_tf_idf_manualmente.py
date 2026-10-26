# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 51: TF-IDF MANUALMENTE
# Bloco 13

import numpy as np
import pandas as pd
corpus = [
    'produto bom rapida entrega',
    'produto ruim quebrado',
    'entrega rapida atendimento bom',
    'atendimento ruim produto bom',
]
vocab = sorted(set(p for doc in corpus for p in doc.split()))
def tf(p, doc):
    palavras = doc.split()
    return palavras.count(p) / len(palavras)
def idf(p, corpus):
    df = sum(1 for doc in corpus if p in doc.split())
    return np.log(len(corpus) / df) if df > 0 else 0
# Calcular matriz TF-IDF
matriz = []
for doc in corpus:
    linha = [tf(p, doc) * idf(p, corpus) for p in vocab]
    matriz.append(linha)
df_tfidf = pd.DataFrame(matriz, columns=vocab)
print(df_tfidf.round(3))
