# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 55: BOW VS TF-IDF: TAREFA REAL
# Bloco 16

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# Hipotético: reviews_corpus e labels (0=neg, 1=pos) já preparados
# 1) BoW
bow = CountVectorizer(ngram_range=(1, 2), min_df=5)
X_bow = bow.fit_transform(reviews_corpus)
score_bow = cross_val_score(
    LogisticRegression(max_iter=1000), X_bow, labels,
    cv=5, scoring='f1'
).mean()
# 2) TF-IDF
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
X_tfidf = tfidf.fit_transform(reviews_corpus)
score_tfidf = cross_val_score(
    LogisticRegression(max_iter=1000), X_tfidf, labels,
    cv=5, scoring='f1'
).mean()
print(f'BoW    F1: {score_bow:.3f}')   # ex: 0.812
print(f'TF-IDF F1: {score_tfidf:.3f}') # ex: 0.834 — geralmente vence
