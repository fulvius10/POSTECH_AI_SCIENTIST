# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula06-v1.pptx
# Slide 53: BERTOPIC EM PT-BR — CÓDIGO COMPLETO SETUP COM SENTENCE TRANSFORMER MULTILINGUAL
# Bloco 03

# pip install bertopic sentence-transformers
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
# Modelo de embeddings em português (multilingual)
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# BERTopic com configuração padrão
topic_model = BERTopic(
    embedding_model=embedder,
    language="multilingual",
    min_topic_size=20,        # ignora tópicos com menos de 20 docs
    nr_topics="auto",
)
# 'reviews' é uma lista de strings em PT
topics, probs = topic_model.fit_transform(reviews)
# Inspecionar tópicos
print(topic_model.get_topic_info().head(10))
# Topic  Count                          Name
#    -1   1245   -1_geral_outros_diversos
#     0    832     0_entrega_prazo_atraso
#     1    721     1_qualidade_produto_bom
#     2    615     2_atendimento_suporte_problema
