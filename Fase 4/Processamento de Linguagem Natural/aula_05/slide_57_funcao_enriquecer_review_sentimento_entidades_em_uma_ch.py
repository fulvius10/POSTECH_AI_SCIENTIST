# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 57: FUNÇÃO ENRIQUECER_REVIEW SENTIMENTO + ENTIDADES EM UMA CHAMADA
# Bloco 16

# 3) NER em paralelo no mesmo texto
nlp = spacy.load('pt_core_news_lg')
def enriquecer_review(texto):
    # Sentiment via pipeline treinado
    sentimento = grid.predict([texto])[0]
    # Entidades via spaCy
    doc = nlp(texto)
    entidades = [(ent.text, ent.label_) for ent in doc.ents]
    return {
        'texto': texto,
        'sentimento': int(sentimento),
        'entidades': entidades
    }
# Uso
resultado = enriquecer_review(
    'Adorei o iPhone 15 da Magazine Luiza, entrega rápida em SP!'
)
print(resultado)
# {'texto': '...', 'sentimento': 1,
#  'entidades': [('iPhone 15', 'MISC'),
#                ('Magazine Luiza', 'ORG'), ('SP', 'LOC')]}
