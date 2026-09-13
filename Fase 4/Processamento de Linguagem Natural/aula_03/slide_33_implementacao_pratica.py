# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 33: IMPLEMENTAÇÃO PRÁTICA
# Bloco 08

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'Adorei o produto, super recomendo',
    'Não gostei muito do produto',
    'Produto bom mesmo, vale o preço',
    'Muito ruim, não comprem',
]
# Comparando ngram_range diferentes
for n_range in [(1, 1), (1, 2), (1, 3), (2, 2)]:
    vec = CountVectorizer(ngram_range=n_range)
    X = vec.fit_transform(corpus)
    print(f'ngram_range={n_range}: {X.shape[1]} features')
# Saída típica:
# ngram_range=(1, 1): 18 features    ← unigrams só
# ngram_range=(1, 2): 35 features    ← +17 bigrams
# ngram_range=(1, 3): 50 features    ← +15 trigrams
# ngram_range=(2, 2): 17 features    ← APENAS bigrams
