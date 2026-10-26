# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula01-v1.pptx
# Slide 55: SPACY: NAMED ENTITY RECOGNITION IDENTIFICANDO PESSOAS, LUGARES E ORGANIZAÇÕES
# Bloco 06

# Identificação de entidades nomeadas em português
texto = ("Leonardo da Vinci nasceu em Vinci, Itália, em 1452. "
        "Foi pintor, cientista e trabalhou para Lorenzo de Médici. "
        "Morreu em Amboise, na França.")
doc = nlp(texto)
print('Entidades encontradas:')
for ent in doc.ents:
    print(f'  {ent.text:25s} → {ent.label_}')
# Saída esperada:
# Leonardo da Vinci         → PER  (pessoa)
# Vinci                     → LOC  (local)
# Itália                    → LOC
# 1452                      → MISC
# Lorenzo de Médici         → PER
# Amboise                   → LOC
# França                    → LOC
