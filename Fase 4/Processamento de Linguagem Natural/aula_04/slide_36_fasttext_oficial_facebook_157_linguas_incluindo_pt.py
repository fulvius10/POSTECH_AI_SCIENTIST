# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula04-v1.pptx
# Slide 36: FASTTEXT OFICIAL (FACEBOOK) 157 LÍNGUAS, INCLUINDO PT
# Bloco 05

# Download do modelo PT (~7 GB)
# wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.pt.bin
# Carregando com Gensim
from gensim.models.fasttext import load_facebook_vectors
modelo = load_facebook_vectors('wiki.pt.bin')
print(modelo['celular'].shape)   # (300,)
print(modelo.most_similar('celular', topn=5))
# [('smartphone', 0.81), ('telefone', 0.78),
#  ('aparelho', 0.71), ('android', 0.68), ('iphone', 0.65)]
# Versão menor (.vec) só com vetores principais ~1.5 GB
# https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.pt.vec
