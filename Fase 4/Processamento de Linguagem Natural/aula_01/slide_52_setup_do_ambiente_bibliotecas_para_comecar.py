# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula01-v1.pptx
# Slide 52: SETUP DO AMBIENTE BIBLIOTECAS PARA COMEÇAR
# Bloco 03

# Instalar bibliotecas
# pip install nltk spacy scikit-learn
# python -m spacy download pt_core_news_lg
import nltk
import spacy
import pandas as pd
from collections import Counter
# Recursos do NLTK em português
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')          # stemmer português
# Carregar modelo do spaCy em português
nlp = spacy.load('pt_core_news_lg')
# Corpus exemplo: reviews fictícias de produtos
corpus = [
    "O produto é excelente, recomendo muito! Chegou rápido.",
    "Péssimo, o aparelho veio com defeito. Não comprem.",
    "Ótima qualidade pelo preço. Entrega rápida e bem embalado.",
    "Adorei! Funciona perfeitamente, superou minhas expectativas.",
    "Esperei 3 horas no telefone. Adorei o atendimento.",  # ironia
]
