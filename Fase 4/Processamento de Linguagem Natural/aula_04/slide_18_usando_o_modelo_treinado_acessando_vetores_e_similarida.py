# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula04-v1.pptx
# Slide 18: USANDO O MODELO TREINADO ACESSANDO VETORES E SIMILARIDADES
# Bloco 03

# Vetor de uma palavra
print(model.wv['produto'][:5])
# Saída: array([0.123, -0.456, 0.789, 0.234, -0.567])
# Tamanho do vetor (vector_size=100)
print(model.wv['produto'].shape)
# Saída: (100,)
# Palavras mais similares
print(model.wv.most_similar('produto', topn=5))
# Saída:
# [('item', 0.78),
#  ('aparelho', 0.74),
#  ('mercadoria', 0.69),
#  ('artigo', 0.65),
#  ('peça', 0.61)]
# Similaridade entre duas palavras
print(model.wv.similarity('produto', 'item'))     # 0.78
print(model.wv.similarity('produto', 'amor'))     # 0.12
# Carregar modelo salvo
model = Word2Vec.load('word2vec_ptbr.model')
