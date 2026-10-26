# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 14: MULTINOMIAL NB + TF-IDF IMPLEMENTAÇÃO EM PIPELINE
# Bloco 02

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Hipotético: 'reviews' é lista de strings, 'labels' é lista 0/1
X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.2, stratify=labels, random_state=42
)
# Pipeline = vetorização + classificador em UM ÚNICO objeto
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)),
    ('clf', MultinomialNB(alpha=1.0)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['negativo', 'positivo']))
