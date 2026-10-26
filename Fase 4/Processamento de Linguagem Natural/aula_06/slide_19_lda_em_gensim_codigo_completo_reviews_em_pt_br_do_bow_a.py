# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula06-v1.pptx
# Slide 19: LDA EM GENSIM — CÓDIGO COMPLETO REVIEWS EM PT-BR, DO BOW AO COHERENCE
# Bloco 01

from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
# 'tokens_limpos' = lista de listas de tokens (output Aula 2)
# Ex.: [["produto", "bom"], ["atendimento", "ruim"], ...]
# 1) Criar dicionário e corpus em BoW
dicionario = corpora.Dictionary(tokens_limpos)
dicionario.filter_extremes(no_below=10, no_above=0.5)
corpus_bow = [dicionario.doc2bow(d) for d in tokens_limpos]
# 2) Treinar LDA com 10 tópicos
lda = LdaModel(corpus=corpus_bow, id2word=dicionario, num_topics=10,
               passes=20, alpha="auto", eta="auto", random_state=42)
# 3) Inspecionar tópicos descobertos
for idx, topic in lda.print_topics(num_words=8):
    print(f"Tópico {idx}: {topic}")
# 4) Coerência (qualidade dos tópicos)
coh = CoherenceModel(model=lda, texts=tokens_limpos,
                     dictionary=dicionario, coherence="c_v")
print(f"Coherence (c_v): {coh.get_coherence():.3f}")
