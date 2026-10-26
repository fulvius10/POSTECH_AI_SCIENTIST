# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 51: SAÍDA DO NER EM PT-BR ENTIDADES RECONHECIDAS PELO SPACY
# Bloco 13

# Saída ilustrativa:
# Entidades encontradas:
#   iPhone 15                 | MISC   | start=21,  end=30
#   Magazine Luiza            | ORG    | start=33,  end=47
#   R$ 5.999                  | MONEY  | start=51,  end=59
#   São Paulo                 | LOC    | start=86,  end=95
#   20 de outubro             | DATE   | start=104, end=117
#   Fedex                     | ORG    | start=124, end=129
#   Apple Brasil              | ORG    | start=148, end=160
# Em uma única chamada:
# - 7 entidades extraídas
# - 4 tipos diferentes (MISC, ORG, MONEY, LOC, DATE)
# - Posições de char para destaque visual
# - Pronto para alimentar dashboard ou banco de dados
# Visualizar em Jupyter:
from spacy import displacy
displacy.render(doc, style='ent')
