# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula04-v1.pptx
# Slide 51: ANALOGIAS EM CÓDIGO "REI ESTÁ PARA HOMEM COMO X ESTÁ PARA MULHER"
# Bloco 09

# A famosa analogia: rei está para homem assim como X está para mulher
# Aritmética: vec(rei) - vec(homem) + vec(mulher) = ?
resultado = modelo.most_similar(
    positive=['rei', 'mulher'],   # adiciona
    negative=['homem'],            # subtrai
    topn=3
)
print(resultado)
# [('rainha', 0.78),
#  ('princesa', 0.72),
#  ('imperatriz', 0.68)]
# Por que funciona?
# A direção homem→mulher no espaço vetorial codifica a relação masculino→feminino
# Aplicada a 'rei', leva ao equivalente feminino: 'rainha'
# Outras analogias:
modelo.most_similar(
    positive=['paris', 'alemanha'],
    negative=['frança'],
    topn=3
)
# [('berlim', 0.74), ('hamburgo', 0.62), ('munique', 0.58)]
