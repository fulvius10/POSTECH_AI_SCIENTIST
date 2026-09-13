# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 33: GRIDSEARCHCV NO PIPELINE TUNING DE VETORIZAÇÃO + CLASSIFICADOR JUNTOS
# Bloco 08

from sklearn.model_selection import GridSearchCV
# Tune AMBOS os componentes do pipeline
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__min_df': [3, 5, 10],
    'tfidf__max_df': [0.85, 0.95],
    'clf__C': [0.1, 1.0, 10.0],
    'clf__penalty': ['l1', 'l2'],
}
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,                  # 5-fold cross-validation
    scoring='f1_macro',    # otimiza F1 macro
    n_jobs=-1,             # usa todos os CPUs
)
grid.fit(X_train, y_train)
print(f'Melhores parâmetros: {grid.best_params_}')
print(f'Melhor F1 macro CV: {grid.best_score_:.3f}')
print(f'Test F1 macro: {grid.score(X_test, y_test):.3f}')
