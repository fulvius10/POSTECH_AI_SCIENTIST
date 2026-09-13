# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula04-v1.pptx
# Slide 31: FASTTEXT RESOLVE OOV NÃO HÁ MAIS "PALAVRA DESCONHECIDA"
# Bloco 04

from gensim.models.fasttext import load_facebook_vectors
# Modelo FastText oficial em português
modelo = load_facebook_vectors('wiki.pt.bin')
# Vetor de palavra comum
v = modelo['amor']
print(v.shape)  # (300,)
# Vetor de NEOLOGISMO que NUNCA esteve no corpus
v_novo = modelo['instagramável']
print(v_novo.shape)  # (300,) — foi inferido pelos subwords!
# Vetor de gíria
v_giria = modelo['baita']
print(v_giria.shape)  # (300,) — funciona
# Em Word2Vec, isso jogaria KeyError. Em FastText, sempre tem vetor.
