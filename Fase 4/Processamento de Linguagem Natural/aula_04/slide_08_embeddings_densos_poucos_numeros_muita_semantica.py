# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula04-v1.pptx
# Slide 8: EMBEDDINGS DENSOS POUCOS NÚMEROS, MUITA SEMÂNTICA
# Bloco 01

# Em embeddings (300 dimensões)
vec_rei      = [0.42, -0.13, 0.78, 0.21, ..., -0.05]   # 300 floats
vec_rainha   = [0.41, -0.18, 0.75, 0.19, ..., -0.07]   # 300 floats
vec_abacaxi  = [-0.31, 0.62, 0.04, -0.55, ..., 0.88]   # 300 floats
# Similaridade do cosseno:
# cos_sim(rei, rainha)  = 0.72   # ALTAMENTE similares!
# cos_sim(rei, abacaxi) = 0.05   # praticamente sem relação
# Espaço vetorial onde GEOMETRIA = SEMÂNTICA
