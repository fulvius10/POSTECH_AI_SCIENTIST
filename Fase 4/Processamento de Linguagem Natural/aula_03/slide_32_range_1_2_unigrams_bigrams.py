# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 32: RANGE (1, 2): UNIGRAMS + BIGRAMS
# Bloco 07

# CountVectorizer com unigrams + bigrams
vec = CountVectorizer(ngram_range=(1, 2), min_df=2)
X = vec.fit_transform(corpus)
# Vocabulário agora inclui AMBOS:
#   ['adorei', 'adorei produto', 'bom', 'entrega',
#    'entrega rápida', 'gostei', 'não', 'não gostei',
#    'produto', 'rápida', ...]
# Em sentiment analysis em PT-BR:
# - Unigrams capturam 'gostei', 'péssimo', 'maravilhoso'
# - Bigrams capturam 'não gostei', 'muito bom', 'super ruim'
print(f'Total de features: {len(vec.get_feature_names_out())}')
