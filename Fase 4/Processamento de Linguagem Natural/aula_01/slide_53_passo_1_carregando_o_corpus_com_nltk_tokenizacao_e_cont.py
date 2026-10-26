# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula01-v1.pptx
# Slide 53: PASSO 1 — CARREGANDO O CORPUS COM NLTK TOKENIZAÇÃO E CONTAGEM
# Bloco 04

from nltk.tokenize import word_tokenize
# Tokenizar todos os documentos
todos_tokens = []
for doc in corpus:
    tokens = word_tokenize(doc.lower(), language='portuguese')
    todos_tokens.extend(tokens)
print(f'Total de tokens: {len(todos_tokens)}')
print(f'Vocabulário (únicos): {len(set(todos_tokens))}')
# Top 10 palavras mais frequentes (Lei de Zipf)
contador = Counter(todos_tokens)
print('\nTop 10 mais frequentes:')
for palavra, freq in contador.most_common(10):
    print(f'  {palavra:15s} → {freq}')
# Saída esperada:
# o, ., e, , o, com, a (stop words dominam)
