# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula04-v1.pptx
# Slide 17: TREINANDO WORD2VEC COM GENSIM SETUP E TREINAMENTO EM PT-BR
# Bloco 02

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)
# Hipotético: 'reviews' é uma lista de strings em PT
# Tokenizamos cada review (já com pré-processamento da Aula 2)
sentences = [word_tokenize(r.lower(), language='portuguese')
             for r in reviews]
# Treina Word2Vec com Skip-gram + negative sampling
model = Word2Vec(
    sentences=sentences,
    vector_size=100,        # dimensionalidade do embedding
    window=5,               # janela de contexto
    min_count=5,            # ignora palavras com freq < 5
    workers=4,              # paralelismo
    sg=1,                   # 1 = Skip-gram, 0 = CBOW
    negative=10,            # negative sampling (10 amostras)
    epochs=10,
)
model.save('word2vec_ptbr.model')
