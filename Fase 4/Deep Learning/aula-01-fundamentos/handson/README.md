# Hands-on — Aula 01: Perceptron, Regressão Logística e XOR

Caso de negócio de referência: **Nubank — aprovação de crédito**.

## O que este notebook investiga

O notebook testa, em dois cenários 2D controlados, o **limite geométrico dos modelos lineares**.
Quatro modelos disputam o mesmo dado:

1. `majoritario` — baseline ingênuo que sempre chuta a classe mais frequente;
2. `perceptron` — o neurônio linear clássico;
3. `logistica` — regressão logística (linear, mas com saída probabilística);
4. `mlp_raso` — MLP mínimo com uma camada escondida e ativação não linear.

O primeiro cenário é **linearmente separável** (`make_blobs`); o segundo é um **XOR com ruído leve**,
construído de propósito para falsificar a hipótese de que "basta treinar melhor o modelo linear".

Além das métricas, o notebook desenha as **fronteiras de decisão**, para você separar visualmente
*erro de otimização* (o modelo poderia acertar, mas não convergiu) de *limitação de representabilidade*
(o modelo nunca vai conseguir, não importa quanto treine).

## Pergunta experimental

> Um classificador linear treinado corretamente consegue resolver qualquer problema binário simples em 2D?

**Hipótese do notebook:** em dados linearmente separáveis, perceptron e logística chegam perto do teto;
em XOR, os lineares ficam presos numa fronteira reta e apenas o MLP raso captura a estrutura correta.

## Como executar

1. Abra `hands-on.ipynb` no VS Code ou no Jupyter.
2. Selecione o kernel do ambiente virtual do curso: `C:\git\aula\.venv`
   (aparece como **"Python (curso IA Generativa)"** / **".venv"**). Sem esse kernel, `scikit-learn`, `pandas` e
   `matplotlib` não são encontrados (o notebook não usa `torch`).
3. Execute **Run All** (ou `Ctrl+Shift+P` → *Run All Cells*). As células têm dependência sequencial —
   rodar fora de ordem quebra a narrativa experimental.

Alternativa por linha de comando:

```bash
PYTHONUTF8=1 JUPYTER_PATH="C:/git/aula/.venv/share/jupyter" \
  C:/git/aula/.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace hands-on.ipynb
```

Não há download de dataset externo: tudo é sintético e gerado dentro do notebook com semente fixa (`SEED = 42`).

## Tempo aproximado

**Menos de 10 segundos** de ponta a ponta (roda tranquilamente só em CPU).

## Saídas esperadas

Com `SEED = 42` os resultados são determinísticos. O notebook exibe duas tabelas e uma matriz de confusão.

**1. Tabela completa dos dois cenários** — a célula do laço de treino faz `display(results_df)` e a tabela
aparece **acima** da figura das fronteiras, na mesma saída:

| dataset | modelo | val_accuracy | test_accuracy | test_balanced_accuracy | test_f1 | test_log_loss | test_roc_auc |
|---|---|---|---|---|---|---|---|
| linear | `perceptron` | 0.991667 | 1.000000 | 1.000000 | 1.000000 | NaN | NaN |
| linear | `logistica` | 0.991667 | 1.000000 | 1.000000 | 1.000000 | 0.019172 | 1.000000 |
| linear | `mlp_raso` | 0.991667 | 1.000000 | 1.000000 | 1.000000 | 0.002331 | 1.000000 |
| linear | `majoritario` | 0.500000 | 0.500000 | 0.500000 | 0.000000 | 18.021827 | 0.500000 |
| xor | `mlp_raso` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.007084 | 1.000000 |
| xor | `logistica` | 0.633333 | 0.725000 | 0.719410 | 0.659794 | 0.692429 | 0.551724 |
| xor | `perceptron` | 0.750000 | 0.700000 | 0.709677 | 0.763158 | NaN | NaN |
| xor | `majoritario` | 0.508333 | 0.516667 | 0.500000 | 0.000000 | 17.421099 | 0.500000 |

**2. Tabela do cenário XOR** — a última célula de código exibe
`results_df.query("dataset == 'xor'")` com as colunas de teste, acima da figura dos erros:

| modelo | test_accuracy | test_f1 | test_log_loss | test_roc_auc |
|---|---|---|---|---|
| `mlp_raso` | **1.000000** | 1.000000 | **0.007084** | 1.000000 |
| `logistica` | **0.725000** | 0.659794 | 0.692429 | 0.551724 |
| `perceptron` | **0.700000** | 0.763158 | NaN | NaN |
| `majoritario` | **0.516667** | 0.000000 | 17.421099 | 0.500000 |

O `NaN` do perceptron não é erro: ele não tem `predict_proba`, então log loss e ROC-AUC não são calculados.

**3. Matriz de confusão do perceptron em XOR (teste)** — impressa logo abaixo da tabela do XOR, na mesma célula:

```
[[26 36]
 [ 0 58]]
```

Linhas são a classe real (0 e 1), colunas a classe prevista. Dos 62 exemplos da classe 0, 36 viram falsos
positivos; dos 58 da classe 1, nenhum falso negativo. É isso que explica o F1 do perceptron (0.763158) ser
maior que o da logística (0.659794) mesmo com acurácia menor (0.700 contra 0.725): o modelo degenerou para
"diga sim para quase todo mundo" (94 dos 120 pontos previstos como classe 1).

Números-chave para levar da aula:

- No cenário linear, os três modelos treináveis empatam em **100% de acurácia no teste** — o perceptron
  cumpre exatamente o que seu teorema promete quando as premissas valem.
- No XOR, a regressão logística cai para **0.725 de acurácia** e, pior, **ROC-AUC de 0.551724** — ou seja,
  quase indistinguível de um chute. A acurácia sozinha esconde o tamanho do fracasso.
- O MLP raso vai a **1.000 de acurácia e log loss 0.007084** no mesmo XOR. A diferença não veio de mais
  épocas nem de melhor otimização: veio da **não linearidade**.
- O baseline majoritário fica em ~0.50 nos dois cenários (0.516667 no XOR) e com **F1 igual a zero** —
  lembrete de que "acurácia de 50%" num problema balanceado significa nenhuma informação.

Além das tabelas, você deve ver três figuras: o *scatter* dos dois datasets de treino, as **fronteiras de decisão**
lado a lado (a fronteira reta da logística cortando o XOR ao meio é a imagem que resume a aula) e o mapa de
**erros no teste** (os "x" da logística concentrados nos quadrantes cruzados; o painel do MLP sem marcações).
