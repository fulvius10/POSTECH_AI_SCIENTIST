# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 57: PIPELINE NLP FINAL TUDO COSTURADO EM UMA FUNÇÃO
# Bloco 15

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
STOP_WORDS_PT = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()
def pipeline_nlp(textos: list[str]) -> list[list[str]]:
    """Lista de textos PT → lista de listas de tokens limpos."""
    resultado = []
    for texto in textos:
        texto = limpar_normalizar(texto)                              # 1
        tokens = word_tokenize(texto, language='portuguese')          # 2
        tokens = [t for t in tokens                                   # 3
                  if t not in STOP_WORDS_PT and len(t) > 2]
        tokens = [stemmer.stem(t) for t in tokens]                    # 4
        resultado.append(tokens)
    return resultado
reviews = ['Adorei o produto, entrega rapidíssima!',
           'Não gostei, veio quebrado e atrasado.']
print(pipeline_nlp(reviews))
# [['ador', 'produt', 'entreg', 'rap'],
#  ['gost', 'vei', 'quebr', 'atras']]
