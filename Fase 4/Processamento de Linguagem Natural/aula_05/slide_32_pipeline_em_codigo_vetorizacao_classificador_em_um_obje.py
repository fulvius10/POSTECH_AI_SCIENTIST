# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 32: PIPELINE EM CÓDIGO VETORIZAÇÃO + CLASSIFICADOR EM UM OBJETO
# Bloco 07

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
])
# Tudo em um único fit/predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
# Salvar o pipeline inteiro
import joblib
joblib.dump(pipe, 'sentiment_model.joblib')
# Carregar e usar em produção
pipe = joblib.load('sentiment_model.joblib')
y_pred = pipe.predict(['Adorei o produto!', 'Péssimo atendimento.'])
# array([1, 0])
