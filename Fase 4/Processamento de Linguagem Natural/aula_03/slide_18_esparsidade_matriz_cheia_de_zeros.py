# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula03-v1.pptx
# Slide 18: ESPARSIDADE MATRIZ CHEIA DE ZEROS
# Bloco 04

# Visualizando a esparsidade
print(f'Shape: {X.shape}')                # (10000, 30000)
print(f'Total de células: {X.shape[0] * X.shape[1]:,}')  # 300,000,000
print(f'Não-zeros: {X.nnz:,}')             # ~1,500,000
print(f'Esparsidade: {1 - X.nnz/(X.shape[0]*X.shape[1]):.2%}')
# Esparsidade: 99.50%
# Armazenar matriz cheia em numpy: ~2.4 GB (300M × 8 bytes)
# Armazenar como esparsa: ~36 MB (1.5M × 24 bytes por trio)
