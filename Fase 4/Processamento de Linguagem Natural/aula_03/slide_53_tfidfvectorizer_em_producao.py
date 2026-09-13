# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 53: TFIDFVECTORIZER EM PRODUÇÃO
# Bloco 15

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),         # unigrams + bigrams
    min_df=5,                   # ignora termos em <5 docs
    max_df=0.85,                # ignora termos em >85% dos docs
    max_features=10000,         # cap no vocabulário
    sublinear_tf=True,          # aplica log(1+TF) em vez de TF puro
)
X = vectorizer.fit_transform(reviews_corpus)  # 10k documentos
print(X.shape)        # (10000, 10000)
print(X.nnz)          # número de elementos não-zero
# Top features por documento
import numpy as np
feature_names = vectorizer.get_feature_names_out()
doc0 = X[0].toarray().flatten()
top_idx = np.argsort(doc0)[-5:][::-1]
print('Top 5 features do doc 0:')
for i in top_idx:
    print(f'  {feature_names[i]}: {doc0[i]:.3f}')
