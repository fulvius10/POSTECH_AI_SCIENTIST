# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 40: CURIOSIDADE POR QUE GPT E BERT TOKENIZAM DE FORMA ESTRANHA
# Bloco 11

# Tokenizer do BERTimbau (BERT em PT-BR)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
print(tok.tokenize('correríamos pela praia'))
# ['corre', '##ría', '##mos', 'pela', 'praia']
print(tok.tokenize('inconstitucionalissimamente'))
# ['in', '##cons', '##ti', '##tu', '##cion', '##ali', '##ss', '##im', '##amente']
