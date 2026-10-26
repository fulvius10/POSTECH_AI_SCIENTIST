# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 32: FAMÍLIA 2B: SPACY O PADRÃO DA INDÚSTRIA MODERNA
# Bloco 08

import spacy
nlp = spacy.load('pt_core_news_lg')
texto = "O Sr. Silva, hoje, tem R$ 1.250,99 no banco!"
# spaCy: tokenização + POS + lema + dependência tudo de uma vez
doc = nlp(texto)
for tok in doc:
    print(f"{tok.text:<12} {tok.lemma_:<12} {tok.pos_}")
# Saída:
# O           o           DET
# Sr.         Sr.         PROPN
# Silva       Silva       PROPN
# ,           ,           PUNCT
# hoje        hoje        ADV
# ...
# 1.250,99    1.250,99    NUM
# no          em o        ADP
# banco       banco       NOUN
# !           !           PUNCT
# 🚀 Vantagem: tokens já vêm com lema, POS, dependência!
