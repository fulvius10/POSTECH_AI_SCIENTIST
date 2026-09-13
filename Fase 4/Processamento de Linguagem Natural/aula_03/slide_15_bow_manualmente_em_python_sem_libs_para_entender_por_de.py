# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 15: BOW MANUALMENTE EM PYTHON SEM LIBS, PARA ENTENDER POR DENTRO
# Bloco 01

# BoW manual — 30 linhas para entender o que sklearn faz
from collections import Counter
corpus = [
    'Adorei o produto, entrega rápida',
    'Produto chegou quebrado',
    'Entrega rápida, atendimento bom',
]
# 1. Tokenizar (simplificado)
docs_tokens = [doc.lower().replace(',','').split() for doc in corpus]
# 2. Construir vocabulário (palavras únicas ordenadas)
vocab = sorted(set(t for doc in docs_tokens for t in doc))
print(f'Vocabulário ({len(vocab)} palavras):', vocab)
# 3. Construir matriz documento-termo
matriz = []
for tokens in docs_tokens:
    contagem = Counter(tokens)
    linha = [contagem.get(p, 0) for p in vocab]
    matriz.append(linha)
import pandas as pd
df = pd.DataFrame(matriz, columns=vocab)
print(df)
