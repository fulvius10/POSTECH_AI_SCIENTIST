# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 16: BOW COM SCIKIT-LEARN COUNTVECTORIZER FAZ TUDO
# Bloco 02

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
corpus = [
    'Adorei o produto, entrega rápida e qualidade boa.',
    'Produto chegou quebrado, qualidade péssima.',
    'Entrega rápida, atendimento bom, recomendo!',
    'Produto bom mas atendimento péssimo.',
]
# CountVectorizer: tokenização + vocabulário + matriz
vectorizer = CountVectorizer(lowercase=True, max_features=20)
X = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names_out()
df = pd.DataFrame(X.toarray(), columns=vocab)
print(df)
print(f'Tamanho do vocabulário: {len(vocab)}')
print(f'Esparsidade: {1 - X.nnz / (X.shape[0] * X.shape[1]):.2%}')
