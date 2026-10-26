# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 17: SAÍDA TÍPICA DO COUNTVECTORIZER MATRIZ DOCUMENTO-TERMO EM AÇÃO
# Bloco 03

# Saída ilustrativa:
#    adorei  atendimento  bom  boa  chegou  entrega  mas  o  ...
# 0       1            0    0    1       0        1    0  1  ...
# 1       0            0    0    0       1        0    0  0  ...
# 2       0            1    1    0       0        1    0  0  ...
# 3       0            1    1    0       0        0    1  0  ...
# Tamanho do vocabulário: 17
# Esparsidade: 73.53%
# Cada linha: um documento
# Cada coluna: uma palavra do vocabulário
# Valores: contagem da palavra no documento
# Cuidado: lowercase=True é o default
# Outros parâmetros úteis:
#   - stop_words: lista para remover
#   - max_features: limita N palavras mais frequentes
#   - min_df / max_df: filtros por frequência
