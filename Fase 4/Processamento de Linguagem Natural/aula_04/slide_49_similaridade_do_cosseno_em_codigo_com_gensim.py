# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula04-v1.pptx
# Slide 49: SIMILARIDADE DO COSSENO EM CÓDIGO COM GENSIM
# Bloco 08

# Similaridade entre duas palavras
print(modelo.similarity('rei', 'rainha'))      # ~0.72
print(modelo.similarity('rei', 'abacaxi'))     # ~0.05
print(modelo.similarity('feliz', 'alegre'))    # ~0.78
print(modelo.similarity('feliz', 'triste'))    # ~0.45 (relacionados, opostos!)
# Palavras mais similares (top-k)
print(modelo.most_similar('celular', topn=5))
# [('smartphone', 0.81),
#  ('telefone', 0.78),
#  ('aparelho', 0.71),
#  ('android', 0.68),
#  ('iphone', 0.65)]
# Em PT-BR
print(modelo.most_similar('feliz', topn=5))
# [('alegre', 0.78), ('contente', 0.74), ('eufórico', 0.71),
#  ('animado', 0.68), ('satisfeito', 0.65)]
