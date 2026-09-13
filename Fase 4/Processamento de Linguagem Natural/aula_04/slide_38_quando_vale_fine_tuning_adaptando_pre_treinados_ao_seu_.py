# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula04-v1.pptx
# Slide 38: QUANDO VALE FINE-TUNING? ADAPTANDO PRÉ-TREINADOS AO SEU DOMÍNIO
# Bloco 06

from gensim.models import Word2Vec, KeyedVectors
# Carrega embeddings pré-treinados
base = KeyedVectors.load_word2vec_format('nilc-w2v-skip-300.txt')
# Inicia novo modelo com os pesos pré-treinados
model = Word2Vec(vector_size=300, min_count=5, sg=1)
model.build_vocab(meu_corpus_dominio)
model.wv.vectors = np.copy(base.vectors)  # transfere pesos
# Continua o treinamento no SEU corpus (fine-tuning)
model.train(meu_corpus_dominio, total_examples=len(meu_corpus_dominio),
            epochs=5)
