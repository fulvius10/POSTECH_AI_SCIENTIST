# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 37: BPE: BYTE-PAIR ENCODING O ALGORITMO DOS TRANSFORMERS
# Bloco 10

# Exemplo simplificado de BPE em ação
# Vocabulário inicial: caracteres individuais
# 'l', 'o', 'w', 'e', 'r', ...
# Iteração 1: 'l' + 'o' aparece muito → mescla → 'lo'
# Iteração 2: 'lo' + 'w' aparece muito → 'low'
# Iteração 3: 'e' + 'r' aparece muito → 'er'
# Após várias iterações: vocabulário tem subwords
# como 'low', 'er', 'est', 'ing', '##ing'
# Resultado para 'lowest':
# ['low', 'est']
# Para palavra rara 'newer':
# ['n', 'ewer'] ou ['new', '##er']
