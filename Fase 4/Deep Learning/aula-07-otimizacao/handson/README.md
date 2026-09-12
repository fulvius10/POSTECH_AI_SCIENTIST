# Hands-on — Aula 07: Otimização e Regularização em um MLP Fixo

Caso de negócio de referência: **Olist — previsão de demanda e tuning de modelo**.

## O que este notebook investiga

Aqui a arquitetura **deixa de ser a variável**. O MLP é fixo do início ao fim e o que muda é o **regime
de treinamento**: otimizador, regularização, taxa de aprendizado e scheduler.

O dataset é o `digits`, mas propositalmente **enfraquecido** para expor overfitting: apenas uma fração
do treino é usada (**420 amostras de treino contra 575 de validação e 360 de teste**) e um **ruído de
rótulo** controlado é injetado numa pequena parcela.

O notebook tem três blocos:

1. **Comparação de otimizadores** — `sgd_base` (SGD puro), `sgd_momentum` e `adamw_cosine`
   (AdamW + weight decay + scheduler cosseno);
2. **Variância entre sementes** — cada configuração roda com 3 sementes (13, 42, 91), e o que se
   reporta é média ± desvio-padrão, não uma execução isolada;
3. **Busca de hiperparâmetros** — grid pequeno e controlado de `lr` x `weight_decay`, sem bibliotecas
   externas de HPO.

A métrica de acompanhamento inclui a **lacuna de generalização** (`train_accuracy − val_accuracy`),
que é o sinal direto do efeito da regularização.

## Pergunta experimental

> Se a arquitetura permanece fixa, escolhas de otimização e regularização alteram de forma relevante
> o desempenho final e a estabilidade do treinamento?

**Hipótese do notebook:** SGD puro converge, mas é sensível à escala do passo; momentum acelera e
reduz oscilação; AdamW com regularização coerente e scheduler simples entrega o melhor equilíbrio
entre desempenho e robustez.

## Como executar

1. Abra `hands-on.ipynb` no VS Code ou no Jupyter.
2. Selecione o kernel do ambiente virtual do curso: `C:\git\aula\.venv`
   (aparece como **"Python (curso IA Generativa)"** / **".venv"**). Sem esse kernel, o `torch` não é encontrado.
3. Execute **Run All**. As comparações finais só fazem sentido se **todas** as configurações de treino
   tiverem rodado, em ordem.

Alternativa por linha de comando:

```bash
PYTHONUTF8=1 JUPYTER_PATH="C:/git/aula/.venv/share/jupyter" \
  C:/git/aula/.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace hands-on.ipynb
```

## Tempo aproximado

**~25 a 30 segundos** — é o notebook mais longo do curso, porque roda 3 configurações x 3 sementes,
mais um grid de 9 combinações de hiperparâmetros.

## Saídas esperadas

Comparação entre regimes de otimização (`summary_df`, média de 3 sementes):

| config | accuracy_mean | accuracy_std | macro_f1_mean | gap_mean |
|---|---|---|---|---|
| `sgd_momentum` | **0.9269** | 0.0085 | 0.9260 | +0.0021 |
| `adamw_cosine` | 0.9167 | **0.0028** | 0.9146 | **−0.0217** |
| `sgd_base` | **0.6324** | **0.0794** | 0.6009 | +0.0221 |

Busca de hiperparâmetros (`search_df`, 9 linhas ordenadas por `val_loss_final`; os três valores de `weight_decay`
de cada `lr` dão a mesma `test_accuracy` — resumo por `lr`):

| lr | test_accuracy | val_loss_final |
|---|---|---|
| 0.010 | **0.9222** | 0.254 |
| 0.003 | 0.8861 | 0.464 |
| 0.001 | 0.8194 | 1.056 |

> Onde cada saída aparece no Run All: a célula de execução exibe, nesta ordem, o `summary_df`, o log do
> scheduler cosseno (`epoch`, `lr`, `val_accuracy` do `adamw_cosine`, 30 linhas), o `search_df` completo e o
> painel de três gráficos. A última célula imprime a configuração vencedora (`best_cfg`) com a acurácia de teste,
> repete o `summary_df` e desenha a matriz de confusão.

Números-chave para levar da aula:

- **O regime de otimização vale mais que a arquitetura.** Com a **mesma rede**, a acurácia vai de
  **0.6324** (SGD puro) a **0.9269** (SGD + momentum): **+29 pontos percentuais** sem mexer numa
  única camada. É a mensagem central da aula.
- **Uma execução só te engana.** O `sgd_base` tem desvio-padrão de **0.0794** entre sementes — quase
  8 pontos percentuais de oscilação. O `adamw_cosine` tem **0.0028**, quase 30x mais estável. Reportar
  uma rodada isolada de SGD puro poderia sugerir 0.55 ou 0.71 conforme a sorte da semente.
- **Regularização aparece no gap, não na acurácia.** O `adamw_cosine` tem acurácia média ligeiramente
  menor que o `sgd_momentum` (0.9167 contra 0.9269), mas a lacuna de generalização é **negativa
  (−0.0217)** — ou seja, a validação está **melhor** que o treino. O `sgd_base`, por outro lado, tem
  gap positivo (+0.0221) *e* desempenho ruim: está sub-treinado e instável ao mesmo tempo.
- **A taxa de aprendizado domina o weight decay.** No grid, mudar `lr` de 0.001 para 0.01 move a
  acurácia de **0.8194 para 0.9222** (+10 pp). Mudar `weight_decay` de 0.0 para 0.0005 mexe na
  `val_loss` na **quarta casa decimal** e **não muda a acurácia de teste**. Priorize o que importa.
- O scheduler cosseno faz o `lr` decair de **0.002992 na época 1 até 0.000000 na época 30**, e a
  `val_accuracy` estabiliza em **0.923478** da época 25 em diante — a curva "assenta" em vez de oscilar.

Figuras esperadas: um painel com três gráficos por configuração (loss de validação, acurácia de validação e
learning rate por época) e a matriz de confusão da melhor configuração do grid (`lr` 0.01, `weight_decay` 0.0,
acurácia de teste 0.9389 com 30 épocas — valores impressos logo acima da matriz).
