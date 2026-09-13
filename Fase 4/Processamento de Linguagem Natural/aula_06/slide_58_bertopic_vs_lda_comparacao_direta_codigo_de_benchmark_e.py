# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula06-v1.pptx
# Slide 58: BERTOPIC VS LDA — COMPARAÇÃO DIRETA CÓDIGO DE BENCHMARK EM CORPUS DE REVIEWS PT-BR
# Bloco 05

import time
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
# 1) LDA via Gensim
t0 = time.time()
lda = LdaModel(corpus=corpus_bow, id2word=dicionario,
               num_topics=10, passes=20, random_state=42)
tempo_lda = time.time() - t0
coh_lda = CoherenceModel(model=lda, texts=tokens_limpos,
                         dictionary=dicionario, coherence="c_v").get_coherence()
# 2) BERTopic
t0 = time.time()
topic_model = BERTopic(embedding_model=embedder,
                       language="multilingual", min_topic_size=20)
topics, _ = topic_model.fit_transform(reviews)
tempo_bertopic = time.time() - t0
# Coerência manual: top words de cada tópico
top_words_bert = [[w for w, _ in topic_model.get_topic(t)[:10]]
                  for t in topic_model.get_topics() if t != -1]
coh_bert = CoherenceModel(topics=top_words_bert, texts=tokens_limpos,
                          dictionary=dicionario, coherence="c_v").get_coherence()
print(f"{'Modelo':<12} {'Tempo (s)':<12} {'Coherence':<12}")
print(f"{'LDA':<12} {tempo_lda:<12.1f} {coh_lda:<12.3f}")
print(f"{'BERTopic':<12} {tempo_bertopic:<12.1f} {coh_bert:<12.3f}")
# Saída em corpus de 10k reviews:
# LDA          45.2          0.428
# BERTopic     312.8         0.567   ← mais coerente, mais lento
