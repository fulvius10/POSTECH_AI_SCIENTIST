# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula01-v1.pptx
# Slide 56: PASSO 3 — PRIMEIRA VETORIZAÇÃO DO TEXTO PRÉ-PROCESSADO A VETORES NUMÉRICOS
# Bloco 07

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stopwords_pt = stopwords.words('portuguese')
# Bag of Words com remoção de stop words
vec = CountVectorizer(stop_words=stopwords_pt)
X = vec.fit_transform(corpus)
print(f'Shape da matriz: {X.shape}')  # (5 docs, N features)
print(f'Vocabulário ({len(vec.vocabulary_)} termos):')
print(vec.get_feature_names_out()[:15])
# Visualizar matriz documento-termo (esparsa!)
import pandas as pd
df_dt = pd.DataFrame(
    X.toarray(),
    columns=vec.get_feature_names_out()
)
print('\nMatriz documento-termo (primeiras colunas):')
print(df_dt.iloc[:, :8])
