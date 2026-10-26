# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 20: PIPELINE COMPLETO: LIMPAR_NORMALIZAR() A FUNÇÃO QUE COSTURA TUDO
# Bloco 05

def limpar_normalizar(texto, remove_acentos=False):
    """Pipeline modular para texto em PT-BR."""
    if not isinstance(texto, str):
        return ''
    texto = texto.lower()                                       # 1
    texto = re.sub(r'<[^>]+>', ' ', texto)                      # 2
    texto = re.sub(r'http\S+|www\.\S+', ' ', texto)             # 3
    texto = re.sub(r'\S+@\S+', ' ', texto)                      # 4
    palavras = [CONTRACTIONS.get(p, p) for p in texto.split()]  # 5
    texto = ' '.join(palavras)
    texto = re.sub(r'[^a-záàâãäéèêëíìîïóòôõöúùûüç\s.,!?]', ' ', texto)  # 6
    if remove_acentos:                                          # 7
        nfkd = unicodedata.normalize('NFKD', texto)
        texto = ''.join(c for c in nfkd if not unicodedata.combining(c))
    texto = re.sub(r'\s+', ' ', texto).strip()                  # 8
    return texto
# Teste
sujo = '<p>Adorei o produto!! https://exemplo.com VC PRECISA TBM!!!</p>'
print(limpar_normalizar(sujo))
# 'adorei o produto !! você precisa também !!!'
