# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 38: BIGRAMS + SENTIMENT ANALYSIS EM PT-BR
# Bloco 10

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# Corpus de reviews + labels (0=neg, 1=pos)
# reviews_corpus = [...]; labels = [...]
# Sem bigrams (unigrams puros)
bow_uni = CountVectorizer(ngram_range=(1, 1), min_df=5)
X_uni = bow_uni.fit_transform(reviews_corpus)
f1_uni = cross_val_score(LogisticRegression(max_iter=1000),
                          X_uni, labels, cv=5, scoring='f1').mean()
# Com unigrams + bigrams
bow_bi = CountVectorizer(ngram_range=(1, 2), min_df=5)
X_bi = bow_bi.fit_transform(reviews_corpus)
f1_bi = cross_val_score(LogisticRegression(max_iter=1000),
                         X_bi, labels, cv=5, scoring='f1').mean()
print(f'Unigrams puros:    F1 = {f1_uni:.3f}')
print(f'Unigrams + bigrams: F1 = {f1_bi:.3f}')
print(f'Ganho: +{(f1_bi - f1_uni)*100:.1f} pontos')
# Tipicamente: ganho de 2-5 pontos em sentiment PT-BR
