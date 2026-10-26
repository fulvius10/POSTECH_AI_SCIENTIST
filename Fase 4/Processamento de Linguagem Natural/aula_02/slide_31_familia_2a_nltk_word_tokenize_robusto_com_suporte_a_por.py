# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 31: FAMÍLIA 2A: NLTK WORD_TOKENIZE ROBUSTO, COM SUPORTE A PORTUGUÊS
# Bloco 07

from nltk.tokenize import word_tokenize, sent_tokenize
texto = "O Sr. Silva, hoje, tem R$ 1.250,99 no banco! Adorei."
# Tokenização de palavras (com language='portuguese')
tokens = word_tokenize(texto, language='portuguese')
print(tokens)
# ['O', 'Sr.', 'Silva', ',', 'hoje', ',', 'tem', 'R$',
#  '1.250,99', 'no', 'banco', '!', 'Adorei', '.']
# ✅ 'Sr.' reconhecido como abreviação
# ✅ Vírgulas separadas como tokens
# ✅ Número 1.250,99 mantido inteiro
# Tokenização de FRASES
frases = sent_tokenize(texto, language='portuguese')
print(frases)
# ['O Sr. Silva, hoje, tem R$ 1.250,99 no banco!', 'Adorei.']
