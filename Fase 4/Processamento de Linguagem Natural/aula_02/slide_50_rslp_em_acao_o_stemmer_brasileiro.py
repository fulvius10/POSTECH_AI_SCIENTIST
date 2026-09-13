# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 50: RSLP EM AÇÃO O STEMMER BRASILEIRO
# Bloco 12

from nltk.stem import RSLPStemmer
import nltk
nltk.download('rslp', quiet=True)
stemmer = RSLPStemmer()
exemplos = ['correndo', 'correu', 'correriam',
            'maravilhoso', 'felicidades', 'menino',
            'meninas', 'casinha', 'gatos']
for palavra in exemplos:
    print(f'{palavra:<15} → {stemmer.stem(palavra)}')
# Saída:
# correndo        → corr
# correu          → corr
# correriam       → corr
# maravilhoso     → maravilh
# felicidades     → felic
# menino          → menin
# meninas         → menin
# casinha         → casinh
# gatos           → gat
