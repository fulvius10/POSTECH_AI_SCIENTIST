# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula01-v1.pptx
# Slide 54: PASSO 2 — ANÁLISE LINGUÍSTICA COM SPACY O PADRÃO DA INDÚSTRIA
# Bloco 05

# spaCy oferece tokenização robusta + POS tagging + lemmatization + NER
doc = nlp("O João comprou um livro na Amazon ontem.")
print(f"{'TOKEN':12s} {'LEMA':12s} {'POS':10s} {'DEP':10s}")
print('-' * 50)
for token in doc:
    print(f"{token.text:12s} {token.lemma_:12s} "
          f"{token.pos_:10s} {token.dep_:10s}")
# Saída esperada:
# TOKEN        LEMA         POS        DEP
# O            o            DET        det
# João         João         PROPN      nsubj
# comprou      comprar      VERB       ROOT
# um           um           DET        det
# livro        livro        NOUN       obj
# na           em o         ADP        case
# Amazon       Amazon       PROPN      obl
# ontem        ontem        ADV        advmod
# .            .            PUNCT      punct
