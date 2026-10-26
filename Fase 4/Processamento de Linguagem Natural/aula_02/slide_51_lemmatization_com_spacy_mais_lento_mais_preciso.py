# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 51: LEMMATIZATION COM SPACY MAIS LENTO, MAIS PRECISO
# Bloco 13

import spacy
nlp = spacy.load('pt_core_news_lg')
texto = 'As crianças estavam correndo felizes pela praia maravilhosa.'
doc = nlp(texto)
for token in doc:
    print(f'{token.text:<14} → {token.lemma_}')
# Saída:
# As             → o
# crianças       → criança
# estavam        → estar
# correndo       → correr
# felizes        → feliz
# pela           → por o
# praia          → praia
# maravilhosa    → maravilhoso
# .              → .
# 🎯 Repare: lemmatization PRESERVA palavras válidas
# 'correr' (não 'corr'), 'maravilhoso' (não 'maravilh')
