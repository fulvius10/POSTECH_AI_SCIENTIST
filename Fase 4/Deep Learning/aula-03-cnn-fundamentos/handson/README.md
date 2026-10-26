# Hands-on — Aula 03: CNNs e o Viés Indutivo Espacial

Caso de negócio de referência: **Embraer — inspeção visual**.

## O que este notebook investiga

Depois de o MLP mostrar utilidade em imagens pequenas (Aula 02), a pergunta natural é: **um viés
indutivo desenhado para visão melhora a eficiência estatística?**

A resposta honesta é **depende dos dados** — e o notebook mostra isso rodando o mesmo par de modelos em
**três cenários**, com o mesmo particionamento, a mesma arquitetura e o mesmo protocolo de treino:

| cenário | dados | o que o cenário testa |
|---|---|---|
| **A · dígitos 8×8** | `load_digits` (1.797 imagens, 10 classes) | imagens minúsculas e **já centralizadas**: a posição de cada traço é fixa |
| **B · formas 28×28 deslocadas** | 6.600 imagens sintéticas, 6 classes, geradas com `numpy` no próprio notebook | a classe depende de um **padrão espacial local** que aparece em **posição sorteada** (±3 px) |
| **controle · formas centradas** | as mesmas 6.600 imagens, com deslocamento **zero** | isola a translação como a variável causal |

Os dois modelos comparados:

- **baseline linear** — regressão logística sobre os pixels achatados (descarta a geometria da imagem);
- **CNN pequena** — três blocos convolucionais terminando em `AdaptiveMaxPool2d(1)`, que guarda apenas a
  **resposta máxima de cada filtro em toda a imagem**. É esse detalhe que dá invariância a translação, e é
  ele que explica tanto a vitória no cenário B quanto a derrota no cenário A.

O cenário B foi desenhado com três cuidados para não virar um espantalho: a **tinta é equalizada** (todo
carimbo soma a mesma intensidade, então a soma de pixels não denuncia a classe), a amplitude/tamanho/
espessura são aleatórios, e o ruído gaussiano é o mesmo para todas as classes.

Há também a visualização dos **mapas de ativação da primeira camada convolucional**, uma comparação de
**recall por classe** entre os dois modelos nos cenários B e controle, e uma galeria dos exemplos que o
baseline erra e a CNN acerta.

> **Por que o dataset mudou.** A versão anterior usava apenas `digits` 8×8. Nessa resolução não há
> estrutura espacial deslocável para a convolução explorar, então a CNN perdia do baseline linear
> (0.9167 contra 0.9556) — o oposto do que a aula ensina. O `digits` foi **mantido como cenário A**
> (é honesto e é didático que o linear vença ali), e o cenário B foi acrescentado para que a aula
> consiga demonstrar **quando** o viés convolucional paga.

## Pergunta experimental

> O viés indutivo da convolução melhora o desempenho em qualquer problema de imagem, ou só quando os
> dados realmente têm estrutura espacial deslocável?

**Hipótese do notebook:** no cenário A o baseline linear vence (os dígitos já vêm alinhados); no cenário B
a CNN vence com folga e com custo paramétrico comparável ou menor; no controle o baseline se recupera,
provando que o problema dele é a translação e não as formas.

> **O experimento confirma a hipótese nos três cenários**, inclusive na parte contraintuitiva: a CNN
> *perde* no `digits`.

## Como executar

1. Abra `hands-on.ipynb` no VS Code ou no Jupyter.
2. Selecione o kernel do ambiente virtual do curso: `C:\git\aula\.venv`
   (aparece como **"Python (curso IA Generativa)"** / **".venv"**). Sem esse kernel, o `torch` não é encontrado.
3. Execute **Run All**. A célula de inspeção dos mapas de ativação depende da célula de treino já ter
   rodado — não pule para o final.

Alternativa por linha de comando:

```bash
PYTHONUTF8=1 JUPYTER_PATH="C:/git/aula/.venv/share/jupyter" \
  C:/git/aula/.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace hands-on.ipynb
```

O `digits` vem embutido no `scikit-learn` e as formas são geradas com `numpy` dentro do notebook —
**nenhum download externo**. Roda em CPU; usa GPU se disponível.

## Tempo aproximado

**~60 segundos** de ponta a ponta em GPU (três cenários × [regressão logística + 25 épocas de CNN]).
Em CPU, conte cerca de 3 minutos — o gargalo é a regressão logística sobre 784 features.

## Saídas esperadas

Comparação final (`comparison_df`):

| cenário | modelo | accuracy | macro_f1 | log_loss | parâmetros treináveis |
|---|---|---|---|---|---|
| A · dígitos 8x8 | Regressão logística | **0.9556** | 0.9555 | **0.1894** | **650** |
| A · dígitos 8x8 | CNN pequena | 0.8611 | 0.8595 | 0.4230 | 4.442 |
| B · formas 28x28 deslocadas | Regressão logística | 0.4402 | 0.4392 | 1.8192 | 4.710 |
| B · formas 28x28 deslocadas | CNN pequena | **0.9803** | **0.9803** | **0.0649** | **4.310** |
| controle · formas centradas | Regressão logística | **0.9985** | 0.9985 | **0.0220** | 4.710 |
| controle · formas centradas | CNN pequena | 0.9826 | 0.9826 | 0.0508 | 4.310 |

Ganho da CNN por cenário (`delta`):

| cenário | acc_linear | acc_cnn | ganho da CNN (pp) |
|---|---|---|---|
| A · dígitos 8x8 | 0.96 | 0.86 | **-9.44** |
| B · formas 28x28 deslocadas | 0.44 | 0.98 | **+54.02** |
| controle · formas centradas | 1.00 | 0.98 | -1.59 |

Números-chave para levar da aula:

- **No cenário B a CNN ganha 54 pontos percentuais gastando menos parâmetros** (4.310 contra 4.710).
  Não é capacidade bruta: é a estrutura certa para o problema. A `log loss` conta a mesma história
  (0.065 contra 1.819, ~28× menor).
- **No cenário A o baseline vence por 9,4 pp com 6,8× menos parâmetros.** Em imagens 8×8 já centralizadas,
  a invariância a translação da CNN **joga fora** informação útil: ali "onde" o pixel está já é o sinal.
- **O controle é a prova causal.** Mesmas formas, mesmo ruído, mesmo gerador — só sem deslocamento.
  O baseline linear salta de **0.4402 para 0.9985**. As formas nunca foram difíceis para ele; o que o
  derrubava era a translação.
- **Recall por classe no cenário B** mostra onde o baseline desmorona: `xadrez` **0.245**, `anel` **0.432**,
  `cruz` **0.432**, `barras verticais` **0.414** — contra **0.959 a 1.000** da CNN. No controle o mesmo
  baseline vai a **0.991–1.000** em todas as classes.
- Existem **719 exemplos de teste (de 1.320)** que o baseline linear erra e a CNN acerta no cenário B.
  A galeria mostra oito deles — todos casos em que a forma simplesmente não estava no lugar "esperado".

Figuras esperadas: grade com amostras dos três cenários, curvas de acurácia de validação da CNN com a
linha do baseline em cada cenário, **mapas de ativação da primeira convolução**, matrizes de confusão
lado a lado (baseline contra CNN no cenário B) e a galeria de acertos exclusivos da CNN.
