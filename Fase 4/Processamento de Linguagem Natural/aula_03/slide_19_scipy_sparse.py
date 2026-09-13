# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 19: SCIPY.SPARSE
# Bloco 05

from scipy.sparse import csr_matrix
import numpy as np
# X já vem como matriz esparsa do CountVectorizer
print(type(X))  # scipy.sparse.csr_matrix
# Usar com modelos de sklearn — todos suportam esparsa nativamente
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)  # ✅ Funciona direto, sem converter
# Converter para denso APENAS se precisar visualizar
# (cuidado: pode estourar memória em corpora grandes!)
X_dense = X.toarray()  # ⚠️ Use só com X pequeno
