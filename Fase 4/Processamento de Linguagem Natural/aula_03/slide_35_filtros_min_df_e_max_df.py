# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 35: FILTROS: MIN_DF E MAX_DF
# Bloco 09

# Filtros para podar n-grams ruidosos ou inúteis
vec = CountVectorizer(
    ngram_range=(1, 2),
    min_df=5,        # ignora termos em MENOS de 5 documentos
    max_df=0.85,     # ignora termos em MAIS de 85% dos documentos
    max_features=10000,  # limita vocabulário aos 10k mais frequentes
)
X = vec.fit_transform(corpus)
# Por que filtrar?
# - min_df remove n-grams raros (typos, palavras únicas) → reduz ruído
# - max_df remove palavras universais (alternativa a stop words)
# - max_features cap absoluto no tamanho do vocabulário
# Exemplo: corpus de 10k docs
# Sem filtros: ~600k features (lento!)
# Com filtros acima: ~10k features (manejável)
