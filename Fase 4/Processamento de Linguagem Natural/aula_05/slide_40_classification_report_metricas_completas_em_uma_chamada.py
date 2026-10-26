# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 40: CLASSIFICATION_REPORT MÉTRICAS COMPLETAS EM UMA CHAMADA
# Bloco 10

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
y_pred = pipe.predict(X_test)
# Report textual completo
print(classification_report(y_test, y_pred,
      target_names=['negativo', 'positivo']))
# Matriz de confusão visual
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
            xticklabels=['neg', 'pos'],
            yticklabels=['neg', 'pos'])
plt.xlabel('Previsto'); plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.savefig('confusion_matrix.png', dpi=150)
