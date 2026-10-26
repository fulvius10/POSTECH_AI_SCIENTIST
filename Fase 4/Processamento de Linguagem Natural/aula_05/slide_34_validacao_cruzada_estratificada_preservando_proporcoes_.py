# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 34: VALIDAÇÃO CRUZADA ESTRATIFICADA PRESERVANDO PROPORÇÕES DE CLASSE
# Bloco 09

from sklearn.model_selection import StratifiedKFold, cross_val_score
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    pipe, X_train, y_train,
    cv=skf,
    scoring='f1_macro',
    n_jobs=-1
)
print(f'F1 macro por fold: {scores}')
print(f'Média: {scores.mean():.3f} ± {scores.std():.3f}')
# Em GridSearchCV, basta passar StratifiedKFold no parâmetro cv
# (o default cv=5 já faz isso para classificação)
