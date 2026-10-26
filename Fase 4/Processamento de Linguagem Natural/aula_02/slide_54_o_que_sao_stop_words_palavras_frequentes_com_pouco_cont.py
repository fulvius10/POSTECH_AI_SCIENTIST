# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 54: O QUE SÃO STOP WORDS? PALAVRAS FREQUENTES COM POUCO CONTEÚDO SEMÂNTICO
# Bloco 14

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)
STOP_WORDS_PT = set(stopwords.words('portuguese'))
print(f'Total: {len(STOP_WORDS_PT)} stop words')
# Total: 207 stop words
print('Algumas:', list(STOP_WORDS_PT)[:15])
# ['o', 'a', 'os', 'as', 'de', 'do', 'da', 'em', 'no', 'na',
#  'que', 'um', 'uma', 'para', 'com', ...]
