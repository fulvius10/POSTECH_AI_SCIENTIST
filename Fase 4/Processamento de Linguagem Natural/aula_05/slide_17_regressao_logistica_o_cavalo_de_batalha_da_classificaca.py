# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 17: REGRESSÃO LOGÍSTICA O CAVALO-DE-BATALHA DA CLASSIFICAÇÃO
# Bloco 04

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ('clf', LogisticRegression(
        max_iter=1000,
        solver='liblinear',     # rápido para texto esparso
        C=1.0,                  # regularização (menor C = mais regularização)
        class_weight='balanced'  # ajusta para classes desbalanceadas
    )),
])
pipe.fit(X_train, y_train)
