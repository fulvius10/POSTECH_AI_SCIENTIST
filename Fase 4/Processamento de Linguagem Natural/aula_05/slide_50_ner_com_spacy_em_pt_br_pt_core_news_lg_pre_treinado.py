# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 50: NER COM SPACY EM PT-BR PT_CORE_NEWS_LG PRÉ-TREINADO
# Bloco 12

import spacy
nlp = spacy.load('pt_core_news_lg')
texto = '''
Ontem comprei um iPhone 15 na Magazine Luiza por R$ 5.999.
A entrega foi feita em São Paulo no dia 20 de outubro pela Fedex.
Atendimento da Apple Brasil foi excelente.
'''
doc = nlp(texto)
print('Entidades encontradas:')
for ent in doc.ents:
    print(f'  {ent.text:<25} | {ent.label_:<6} | '
          f'start={ent.start_char}, end={ent.end_char}')
# Acesso programático: doc.ents é tupla de Span
# Cada Span tem .text, .label_, .start_char, .end_char,
# .start (token), .end (token), .root, etc.
