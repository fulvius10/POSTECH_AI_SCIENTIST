# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 15: LIMPEZA: HTML, URLS E CONTRAÇÕES
# Bloco 03

# Mapa de contrações comuns em PT-BR informal
CONTRACTIONS = {
    'vc': 'você', 'tb': 'também', 'tbm': 'também',
    'pq': 'por que', 'obg': 'obrigado', 'blz': 'beleza',
    'td': 'tudo', 'msm': 'mesmo', 'vlw': 'valeu',
    'n': 'não', 'naum': 'não', 'cmprar': 'comprar',
    'mt': 'muito', 'pra': 'para'
}
def limpar_texto(texto):
    texto = texto.lower()                                  # 1. Lowercase
    texto = re.sub(r'<[^>]+>', ' ', texto)                 # 2. HTML
    texto = re.sub(r'http\S+|www\.\S+', ' ', texto)        # 3. URLs
    texto = re.sub(r'\S+@\S+', ' ', texto)                 # 4. E-mails
    # 5. Expande contrações
    palavras = texto.split()
    palavras = [CONTRACTIONS.get(p, p) for p in palavras]
    return ' '.join(palavras)
print(limpar_texto(reviews[0]))
# Saída: 'adorei o produto!! você precisa comprar também 😍'
