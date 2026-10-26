# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 12: SUAVIZAÇÃO DE LAPLACE EVITA PROBABILIDADES ZERO
# Bloco 01

# Suavização de Laplace (add-one smoothing)
# Adiciona alpha (default=1.0) ao numerador,
# e alpha * |vocab| ao denominador
# Antes da suavização:
# P('quebrado' | positivo) = 0 / 1000 = 0  ← problema
# Depois da suavização (alpha=1):
# P('quebrado' | positivo) = (0 + 1) / (1000 + |vocab|)
#                          = 1 / 31000
#                          ≈ 0.000032  ← pequeno mas não zero
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0)  # alpha = força da suavização
