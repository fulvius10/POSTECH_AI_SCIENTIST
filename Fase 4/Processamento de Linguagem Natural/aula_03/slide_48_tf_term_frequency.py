# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 48: TF: TERM FREQUENCY
# Bloco 11

def tf(palavra, doc):
    """Term Frequency normalizado pelo tamanho do doc."""
    palavras = doc.split()
    return palavras.count(palavra) / len(palavras)
doc = 'o produto bom o produto rápido'
# 6 palavras totais
print(tf('o', doc))         # 2/6 = 0.333
print(tf('produto', doc))   # 2/6 = 0.333
print(tf('bom', doc))       # 1/6 = 0.167
print(tf('rápido', doc))    # 1/6 = 0.167
