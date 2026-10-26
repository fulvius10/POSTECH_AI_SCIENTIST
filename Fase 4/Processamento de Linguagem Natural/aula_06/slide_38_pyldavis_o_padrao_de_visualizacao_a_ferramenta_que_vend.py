# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula06-v1.pptx
# Slide 38: PYLDAVIS — O PADRÃO DE VISUALIZAÇÃO A FERRAMENTA QUE VENDEU LDA A CIENTISTAS DE DADOS
# Bloco 02

# pip install pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
# Preparar visualização (lda, corpus_bow, dicionario do slide anterior)
vis = gensimvis.prepare(lda, corpus_bow, dicionario)
pyLDAvis.save_html(vis, "topicos_pyldavis.html")
# Abrir o HTML no browser para exploração interativa.
