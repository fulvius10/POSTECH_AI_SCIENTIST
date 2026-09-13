# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 18: INTERPRETABILIDADE DOS COEFICIENTES A GRANDE VANTAGEM DA LR
# Bloco 05

# Acessando coeficientes do modelo treinado
import numpy as np
vocab = pipe.named_steps['tfidf'].get_feature_names_out()
coefs = pipe.named_steps['clf'].coef_[0]
# Top 10 palavras mais POSITIVAS (coef alto)
top_pos = np.argsort(coefs)[-10:][::-1]
print('Mais associadas a POSITIVO:')
for i in top_pos:
    print(f'  {vocab[i]:<20s} ({coefs[i]:+.3f})')
# Saída:
#   adorei         (+2.450)
#   excelente      (+2.103)
#   maravilhoso    (+1.987)
#   recomendo      (+1.812)
#   perfeito       (+1.654)
# Top 10 NEGATIVAS
top_neg = np.argsort(coefs)[:10]
print('Mais associadas a NEGATIVO:')
# pessimo, ruim, quebrado, defeituoso, horrivel, ...
