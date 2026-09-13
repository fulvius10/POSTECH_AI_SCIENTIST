# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 14: SETUP DO AMBIENTE
# Bloco 02

import re
import unicodedata
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
# Recursos NLTK em português
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('rslp', quiet=True)
# Modelo spaCy em português
# python -m spacy download pt_core_news_lg
nlp = spacy.load('pt_core_news_lg')
# Corpus exemplo: reviews sujas reais
reviews = [
    "<p>ADOREI o produto!! vc precisa cmprar tbm 😍</p>",
    "Pessimo!! veio quebrado >:( https://reclame.com",
    "Mt bom mesmo, entrega rapida! 5 estrelas pra esse vendedor.",
    "Não gostei :( voltei o item, atendimento foi pessimo tbm.",
]
