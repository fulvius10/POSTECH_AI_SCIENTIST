# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 56: PIPELINE COMPLETO TF-IDF + LOGREG + NER EM PARALELO
# Bloco 15

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import spacy
# 1) Modelo de classificação de sentimento
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
])
# 2) GridSearch para tuning
param_grid = {
    'tfidf__min_df': [3, 5, 10],
    'tfidf__max_df': [0.85, 0.95],
    'clf__C': [0.1, 1.0, 10.0],
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid.fit(reviews_train, labels_train)
print(f'Melhores parâmetros: {grid.best_params_}')
print(f'Melhor F1 macro: {grid.best_score_:.3f}')
