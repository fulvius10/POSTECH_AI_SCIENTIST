# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 49: IDF: INVERSE DOCUMENT FREQUENCY
# Bloco 12

import numpy as np
def idf(palavra, corpus):
    """Inverse Document Frequency."""
    df = sum(1 for doc in corpus if palavra in doc.split())
    return np.log(len(corpus) / df) if df > 0 else 0
corpus = [
    'o produto bom',
    'o produto ruim',
    'o produto quebrado',
    'o produto rápido',
]
print(idf('o', corpus))         # log(4/4) = 0     ← em TODO doc
print(idf('produto', corpus))   # log(4/4) = 0     ← em TODO doc
print(idf('bom', corpus))       # log(4/1) = 1.39  ← só em 1 doc
