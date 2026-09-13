# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula06-v1.pptx
# Slide 54: BERTOPIC — VISUALIZAÇÕES NATIVAS INSPIRADAS NO PYLDAVIS E MAIS RICAS AINDA
# Bloco 04

# Mapa de tópicos em 2D (similar ao pyLDAvis)
fig = topic_model.visualize_topics()
fig.write_html("mapa_topicos.html")
# Barras horizontais com palavras-chave por tópico
fig = topic_model.visualize_barchart(top_n_topics=10)
fig.write_html("barras_topicos.html")
# Hierarquia de tópicos (dendrograma)
fig = topic_model.visualize_hierarchy(top_n_topics=20)
fig.write_html("hierarquia.html")
# Heatmap de similaridade entre tópicos
fig = topic_model.visualize_heatmap(top_n_topics=15)
fig.write_html("heatmap.html")
# Essas visualizações são especialmente úteis em apresentações
# para times de produto, marketing ou C-level que precisam
# entender o que está sendo discutido nos dados.
