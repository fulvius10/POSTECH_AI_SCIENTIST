# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula01-v1.pptx
# Slide 30: POLISSEMIA EM CÓDIGO
# Bloco 01

# A palavra "banco" tem múltiplos sentidos em português
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    "O banco aprovou meu empréstimo ontem",       # financeiro
    "Sentei no banco da praça para descansar",     # móvel
    "O banco de dados PostgreSQL caiu novamente",  # tech
]
vec = CountVectorizer()
X = vec.fit_transform(corpus)
print('Vocabulário:', vec.get_feature_names_out())
# Para o BoW, 'banco' aparece igual nos 3 documentos.
# Não há nada que distinga sentido financeiro, móvel ou tech!
print('Matriz documento-termo:')
print(X.toarray())
# A coluna de 'banco' tem 1 em todas as linhas → polissemia perdida
