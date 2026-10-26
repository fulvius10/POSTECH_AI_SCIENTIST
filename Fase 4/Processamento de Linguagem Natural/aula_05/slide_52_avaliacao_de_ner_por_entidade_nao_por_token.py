# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 52: AVALIAÇÃO DE NER POR ENTIDADE, NÃO POR TOKEN
# Bloco 14

# pip install seqeval
from seqeval.metrics import classification_report as ner_report
# y_true e y_pred são listas de listas de tags BIO
# Exemplo:
# y_true[0] = ['O', 'O', 'B-PER', 'I-PER', 'O', 'B-LOC']
# y_pred[0] = ['O', 'O', 'B-PER', 'I-PER', 'O', 'B-LOC']
print(ner_report(y_true, y_pred))
# Saída ilustrativa:
#               precision    recall  f1-score   support
#          LOC       0.92      0.89      0.90       150
#          ORG       0.85      0.81      0.83       200
#          PER       0.94      0.93      0.93        80
#    micro avg       0.89      0.86      0.87       430
#    macro avg       0.90      0.88      0.89       430
