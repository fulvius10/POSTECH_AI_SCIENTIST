# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 33: SENTENCE TOKENIZATION QUEBRANDO TEXTO EM FRASES
# Bloco 09

# Tokenização de FRASES (não palavras)
from nltk.tokenize import sent_tokenize
texto = ("Adorei o produto. Chegou rápido! "
        "Recomendo a todos. Mas o preço foi alto...")
frases = sent_tokenize(texto, language='portuguese')
for i, f in enumerate(frases, 1):
    print(f'{i}. {f}')
# 1. Adorei o produto.
# 2. Chegou rápido!
# 3. Recomendo a todos.
# 4. Mas o preço foi alto...
# Em spaCy é automático
for sent in nlp(texto).sents:
    print(sent.text)
