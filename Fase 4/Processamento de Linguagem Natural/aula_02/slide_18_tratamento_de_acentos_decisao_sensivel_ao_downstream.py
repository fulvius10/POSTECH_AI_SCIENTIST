# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 18: TRATAMENTO DE ACENTOS DECISÃO SENSÍVEL AO DOWNSTREAM
# Bloco 04

import unicodedata
def remove_acentos(texto):
    """Remove acentos: 'maravilhoso' → 'maravilhoso'.
    Útil para busca; perigoso para NER."""
    nfkd = unicodedata.normalize('NFKD', texto)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))
# Exemplos
print(remove_acentos('São Paulo'))      # 'Sao Paulo'
print(remove_acentos('experiência'))    # 'experiencia'
print(remove_acentos('coração'))        # 'coracao'
