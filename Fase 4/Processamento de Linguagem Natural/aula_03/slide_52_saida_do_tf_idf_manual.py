# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 52: SAÍDA DO TF-IDF MANUAL
# Bloco 14

# Saída ilustrativa (arredondada):
#                atendimento    bom  entrega  produto  quebrado  rapida  ruim
# Doc 1               0.000     0.072  0.072    0.072     0.000   0.072  0.000
# Doc 2               0.000     0.000  0.000    0.096     0.462   0.000  0.231
# Doc 3               0.173     0.072  0.072    0.000     0.000   0.072  0.000
# Doc 4               0.173     0.072  0.000    0.072     0.000   0.000  0.231
# Repare como cada palavra recebe peso DIFERENTE em cada doc:
# - 'produto' aparece em 3/4 docs → IDF baixo → peso BAIXO
# - 'quebrado' aparece em 1/4 docs → IDF ALTO → peso 0.462 (TOP!)
# - 'atendimento' aparece em 2/4 docs → IDF médio
# A palavra MAIS DISCRIMINATIVA do Doc 2 é 'quebrado' — exatamente o que faz
# sentido! É a palavra que SINALIZA o problema do produto.
