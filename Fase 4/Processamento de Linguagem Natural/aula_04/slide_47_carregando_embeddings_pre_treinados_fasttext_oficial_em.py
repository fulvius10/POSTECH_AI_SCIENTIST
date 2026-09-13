# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula04-v1.pptx
# Slide 47: CARREGANDO EMBEDDINGS PRÉ-TREINADOS FASTTEXT OFICIAL EM PT-BR
# Bloco 07

from gensim.models.fasttext import load_facebook_vectors
# Baixe antes:
# wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.pt.bin
# Carregando o modelo (pode demorar ~30s, ocupa ~7GB de RAM)
modelo = load_facebook_vectors('wiki.pt.bin')
# Vetor de uma palavra
v = modelo['amor']
print(v.shape)  # (300,)
print(v[:5])    # array([0.123, -0.456, 0.789, 0.234, -0.567])
# Funciona até com OOV graças a subwords:
v_neologismo = modelo['instagramável']
print(v_neologismo.shape)  # (300,) - foi inferido pelos subwords
# Tamanho do vocabulário
print(f'Vocabulário: {len(modelo.key_to_index):,} palavras')
# ~2 milhões de palavras conhecidas + qualquer OOV via subword
