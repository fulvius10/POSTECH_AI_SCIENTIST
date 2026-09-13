# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula05-v1.pptx
# Slide 49: BIO TAGGING O ESQUEMA PADRÃO PARA ANOTAÇÃO DE NER
# Bloco 11

# Exemplo: "Maria Silva trabalha em São Paulo"
# Tokens:    [Maria]   [Silva]   [trabalha]  [em]  [São]    [Paulo]
# Tags BIO:  [B-PER]   [I-PER]   [O]         [O]   [B-LOC]  [I-LOC]
# Como ler:
# - 'Maria' começa entidade do tipo PER (B-PER)
# - 'Silva' continua a mesma entidade PER (I-PER)
# - 'trabalha', 'em' não são entidades (O)
# - 'São' começa entidade do tipo LOC (B-LOC)
# - 'Paulo' continua a mesma entidade LOC (I-LOC)
# Variantes: BIOES (adiciona E=End, S=Single)
# BIO é o padrão por simplicidade
