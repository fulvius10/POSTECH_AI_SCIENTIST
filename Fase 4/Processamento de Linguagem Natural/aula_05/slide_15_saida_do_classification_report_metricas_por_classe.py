# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 15: SAÍDA DO CLASSIFICATION_REPORT MÉTRICAS POR CLASSE
# Bloco 03

# Saída ilustrativa do classification_report:
#               precision    recall  f1-score   support
#     negativo       0.85      0.82      0.83      1000
#     positivo       0.83      0.86      0.84      1000
#     accuracy                           0.84      2000
#    macro avg       0.84      0.84      0.84      2000
# weighted avg       0.84      0.84      0.84      2000
# Como ler:
# - precision: das previsões positivas, quantas certas?
# - recall: dos positivos reais, quantos encontramos?
# - f1-score: harmônica de precision e recall
# - support: número de exemplos por classe
# - macro avg: média simples (todas classes pesam igual)
# - weighted avg: ponderada pelo support
