# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 31: TRIGRAMS (N=3)
# Bloco 06

# Trigrams capturam expressões idiomáticas e contexto mais longo
from sklearn.feature_extraction.text import CountVectorizer
textos = [
    'não gostei nada do produto',
    'muito bom mesmo, recomendo',
    'foi de jeito nenhum',
]
# ngram_range=(3, 3) → APENAS trigrams
vec = CountVectorizer(ngram_range=(3, 3))
X = vec.fit_transform(textos)
print(vec.get_feature_names_out())
# Saída:
# ['de jeito nenhum', 'do produto'... 'mesmo recomendo',
#  'muito bom mesmo', 'não gostei nada']
# Trigrams capturam:
# - Expressões fixas: 'de jeito nenhum'
# - Modificadores intensivos: 'muito bom mesmo'
# - Negações compostas: 'não gostei nada'
