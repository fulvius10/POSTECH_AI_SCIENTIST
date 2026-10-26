# Hands-on — Aula 06: Atenção e Transformer Encoder em Reviews em Português

Caso de negócio de referência: **Mercado Livre — entendimento de reviews**.

## O que este notebook investiga

O notebook confronta duas hipóteses de modelagem para classificação textual em português:

- **baselines lexicais** — TF-IDF + regressão logística, em **duas versões**: só unigramas (bag-of-words
  puro) e 1–2 gramas (o remendo usual para recuperar um pouco de ordem local);
- **mini-Transformer encoder** — autoatenção treinada do zero, com token `<cls>`, codificação posicional
  senoidal e pipeline de tokenização explícito (sem modelo pré-treinado, sem download de pesos), rodado
  com **1 e com 2 camadas**.

A tarefa é dizer **qual aspecto da compra motivou a reclamação** (entrega, qualidade ou atendimento) ou
reconhecer que a review é um elogio. O corpus tem **3.200 reviews sintéticas em 4 classes** e é construído
para que **a ordem e o contexto importem**: todas as classes compartilham o mesmo vocabulário de
julgamentos (`decepcionou`, `surpreendeu`, `falhou feio`, `funcionou direito`...) e o que liga um
julgamento ao aspecto certo é posição, escopo e ordem — nunca a presença isolada de uma palavra.

Cinco construções linguísticas:

| construção | exemplo | por que é difícil para bag-of-words |
|---|---|---|
| **direta** | `desempenho no dia a dia estragou a compra` | não é difícil — é o piso comum |
| **adversativa** | `dessa vez, acabamento, na prática, agradou bastante, mas sem exagero, garantia, pra mim, falhou feio` | a sacola de palavras é **idêntica** se trocarmos qual aspecto foi elogiado e qual foi criticado |
| **negação de escopo** | `não dá pra dizer que garantia de verdade agradou bastante` (= reclamação) contra `não dá pra dizer que suporte até agora decepcionou` (= elogio) | a classe é o **XOR** entre "há negação" e "polaridade do julgamento" — um produto que a regressão logística linear sobre contagens não forma |
| **anáfora ordinal** | `comparando sem exagero, acabamento, no dia a dia, e dessa vez, suporte, até agora, só o segundo decepcionou` | é preciso ligar `o segundo` ao aspecto que apareceu em segundo lugar, a muitos tokens de distância |
| **ambígua** (10% das reclamações) | `dessa vez, chat, depois de duas semanas, decepcionou, e em geral, rastreio, de verdade, falhou feio` | duas críticas coordenadas por `e`: o texto **não diz** qual foi a principal |

Dois cuidados de desenho tornam o experimento honesto:

1. **Fillers obrigatórios** (`de verdade`, `no dia a dia`, `dessa vez`, ...) antes e depois de **cada**
   aspecto. Sem isso, o TF-IDF com bigramas resolveria a tarefa pelo atalho `mas entrega`.
2. **Negação equilibrada**: `não`, `ninguém`, `exagero` aparecem tanto em elogios quanto em reclamações.
   A palavra de negação sozinha é inútil; o que importa é sobre o que ela incide.

A construção **ambígua** cria um **erro de Bayes de ~3,6 pontos** (teto teórico **0.964**). Nenhum modelo
pode passar disso, porque a informação não está no texto — é o antídoto contra a leitura ingênua de que
"acurácia alta o bastante resolve".

Ao final há a visualização dos **pesos de atenção** — incluindo a linha do token `<cls>`, isto é, o que o
classificador efetivamente consultou — tratados como instrumento diagnóstico, **não** como explicação causal.

> **Por que o corpus mudou.** Na versão anterior cada classe tinha vocabulário próprio e fortemente
> discriminativo (`ótimo`, `confiável` de um lado; `confuso`, `lento` do outro). TF-IDF e Transformer
> chegavam **ambos a acurácia 1.000** e a tabela de erros saía **vazia**: o experimento não conseguia
> distinguir as hipóteses. A estrutura didática foi mantida; o que mudou foi a **dificuldade linguística
> do corpus**.

## Pergunta experimental

> Quando a classe depende de *qual* aspecto foi criticado — e não de *quais palavras* aparecem —, um
> encoder com autoatenção supera um baseline bag-of-words? E adicionar bigramas ao TF-IDF é suficiente
> para fechar a diferença?

**Hipótese do notebook:** o TF-IDF acerta as frases diretas e cai para perto de 50% nas adversativas e
ordinais; bigramas não resolvem porque a dependência é mais longa que dois tokens; uma camada de atenção
resolve adversativa e ordinal mas tropeça na negação de escopo (que é uma composição em dois passos);
duas camadas resolvem também a negação; e ninguém passa de ~50% nas frases ambíguas.

> **O experimento confirma todas essas previsões.**

## Como executar

1. Abra `hands-on.ipynb` no VS Code ou no Jupyter.
2. Selecione o kernel do ambiente virtual do curso: `C:\git\aula\.venv`
   (aparece como **"Python (curso IA Generativa)"** / **".venv"**). Sem esse kernel, o `torch` não é encontrado.
3. Execute **Run All**. A visualização de atenção depende de as células de treino já terem rodado.

Alternativa por linha de comando:

```bash
PYTHONUTF8=1 JUPYTER_PATH="C:/git/aula/.venv/share/jupyter" \
  C:/git/aula/.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace hands-on.ipynb
```

O corpus é gerado localmente — nenhum dataset externo é baixado.

## Tempo aproximado

**~30 segundos** de ponta a ponta em GPU (dois baselines TF-IDF + dois encoders de 30 épocas).
Em CPU, cerca de 2 minutos.

## Saídas esperadas

Corpus: **3.200 reviews**, treino 1.920 / validação 640 / teste 640; **7,2% ambíguas**; vocabulário de
**95 tokens**; comprimento máximo **34 tokens**.

Comparação final (`comparison_df`):

| modelo | accuracy | macro_f1 | log_loss | features / parâmetros |
|---|---|---|---|---|
| TF-IDF unigramas + LR | 0.7453 | 0.7420 | 0.6080 | 88 features |
| TF-IDF 1-2 gramas + LR | 0.7344 | 0.7338 | 0.7220 | 971 features |
| Mini-Transformer (1 camada) | 0.8531 | 0.8541 | 0.3320 | 39.812 |
| **Mini-Transformer (2 camadas)** | **0.9359** | **0.9354** | **0.2354** | 73.284 |
| *teto de Bayes do corpus* | *0.964* | — | — | — |

Acurácia por construção (`by_construction`):

| construção | n teste | TF-IDF uni | TF-IDF 1–2 | atenção 1 camada | atenção 2 camadas |
|---|---|---|---|---|---|
| direta | 169 | 1.000 | 0.994 | 0.988 | **1.000** |
| adversativa | 184 | 0.668 | 0.679 | 0.875 | **0.984** |
| negação de escopo | 110 | 0.700 | 0.755 | 0.700 | **0.936** |
| anáfora ordinal | 124 | 0.661 | 0.605 | 0.911 | **0.952** |
| ambígua | 53 | 0.491 | 0.358 | 0.528 | 0.528 |

Números-chave para levar da aula:

- **A atenção ganha 19,1 pontos percentuais** do melhor baseline lexical (0.9359 contra 0.7453) — e
  **nada satura**: o melhor modelo fica 2,8 pontos abaixo do teto de Bayes do próprio corpus.
- **Adicionar bigramas não ajuda: piora.** 0.7344 contra 0.7453 do unigrama, e a `log loss` sobe de
  0.608 para 0.722. Como os fillers garantem distância mínima entre aspecto e julgamento, a janela de
  dois tokens não alcança a dependência e só acrescenta 883 features ruidosas. É a evidência direta de
  que o problema **não é falta de ordem local**, é falta de contexto.
- **A profundidade importa de forma previsível.** A segunda camada resolve **76 casos** que a primeira
  erra, e **32 deles são de negação de escopo** — a construção que exige compor duas informações em
  etapas. Note que a atenção com 1 camada fica em **0.700** na negação, exatamente empatada com o
  TF-IDF unigrama: ter atenção não basta, é preciso poder **compor** atenção.
- **A tabela de erros não sai vazia.** Dos 640 exemplos de teste: **142** só a atenção acerta, **13**
  só o TF-IDF acerta e **28** ambos erram. Os casos "só a atenção acerta" são majoritariamente de
  negação, adversativa e anáfora ordinal — exatamente onde a hipótese previa.
- **O erro residual é do problema, não do modelo.** Dos **41 erros** da atenção com 2 camadas,
  **25 são da construção ambígua**. Descontando-os, o erro "de modelo" cai para 16 em 587 (**2,7%**).
  Separar erro de modelo de erro de rótulo é parte do trabalho de quem avalia.

Figuras esperadas: curvas de loss e de acurácia de validação das duas profundidades com as linhas dos
baselines, matrizes de confusão lado a lado (TF-IDF 1–2 gramas contra atenção de 2 camadas) e o
**heatmap dos pesos de atenção** de uma frase de negação, acompanhado do barplot do que o token `<cls>`
consultou.
