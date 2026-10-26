# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 30: FAMÍLIA 1: SPLIT() INGÊNUO O BASELINE DIDÁTICO
# Bloco 06

# split() simples — divide só por espaços
texto = "O Sr. Silva, hoje, tem R$ 1.250,99 no banco!"
tokens = texto.split()
print(tokens)
# ['O', 'Sr.', 'Silva,', 'hoje,', 'tem', 'R$', '1.250,99', 'no', 'banco!']
# Problemas:
# - 'Silva,' (com vírgula) ≠ 'Silva' → tokens DIFERENTES no vocabulário
# - 'banco!' (com !) ≠ 'banco' → tokens DIFERENTES
# - 'hoje,' não separa a vírgula
# Útil só como BASELINE didático.
# Em produção: NUNCA use split() puro.
