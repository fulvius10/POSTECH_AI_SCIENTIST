# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 29: CLASS_WEIGHT='BALANCED' LIDANDO COM CLASSES DESBALANCEADAS
# Bloco 06

# Sem class_weight: modelo prevê tudo como majoritária
LogisticRegression()
# Recall na classe minoritária: ~5%  ← inútil
# Com class_weight='balanced': pesos inversos à frequência
LogisticRegression(class_weight='balanced')
# Recall na classe minoritária: ~75%  ← funcional!
# Equivalente:
# class_weight={0: 1.0, 1: 19.0}  # se 1 é 5% dos dados
# Outras alternativas:
# - Oversampling: SMOTE da imblearn
# - Undersampling: RandomUnderSampler
# - Modelos com built-in support: XGBoost com scale_pos_weight
