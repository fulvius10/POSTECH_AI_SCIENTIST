# Hands-on — Aula 02: MLP, Backpropagation e Fluxo de Gradiente

Caso de negócio de referência: **TIM — previsão de churn**.

## O que este notebook investiga

Saímos do problema geométrico 2D da Aula 01 e vamos para um problema **multiclasse real**: o dataset
`digits` do `scikit-learn` (imagens 8x8 em escala de cinza, 10 classes, ~1800 amostras).

O notebook compara:

- **baseline linear** — regressão logística sobre os 64 pixels achatados;
- **MLP raso** — rede com camadas escondidas treinada por backpropagation em PyTorch.

O diferencial da prática não é só a métrica final: a cada época o notebook registra `cross-entropy`,
acurácia, macro-F1 **e as normas do gradiente da primeira e da última camada**. É esse último par que
permite ver, na prática, se o sinal de aprendizado está chegando às camadas iniciais ou morrendo no caminho.

Ao final há uma inspeção dos erros residuais: quais dígitos continuam ambíguos mesmo depois do treino.

## Pergunta experimental

> Em um problema visual multiclasse pequeno, um MLP já produz ganho mensurável sobre um baseline
> linear quando treinado com backpropagation?

**Hipótese do notebook:** a logística é um baseline forte (o `digits` é bem comportado); o MLP deve
superá-la ao modelar combinações não lineares entre pixels; e, se o gradiente da primeira camada se
mantiver estável, a curva de validação melhora sem colapsar em overfitting precoce.

## Como executar

1. Abra `hands-on.ipynb` no VS Code ou no Jupyter.
2. Selecione o kernel do ambiente virtual do curso: `C:\git\aula\.venv`
   (aparece como **"Python (curso IA Generativa)"** / **".venv"**). Sem esse kernel, o `torch` não é encontrado.
3. Execute **Run All**. A reprodutibilidade depende de rodar do topo, em ordem (`SEED = 42` controla
   particionamento, inicialização e shuffling dos lotes).

Alternativa por linha de comando:

```bash
PYTHONUTF8=1 JUPYTER_PATH="C:/git/aula/.venv/share/jupyter" \
  C:/git/aula/.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace hands-on.ipynb
```

O `digits` já vem embutido no `scikit-learn` — não há download externo. O notebook usa GPU se houver
(`device` sai como `'cuda'`), mas roda em CPU sem problema.

## Tempo aproximado

**~15 segundos** de ponta a ponta (30 épocas de treino em um MLP pequeno).

## Saídas esperadas

A célula de treino exibe, acima dos gráficos, três tabelas: o `comparison_df` (comparação final no teste),
o `history_df.head()` (épocas 1–5) e o `grad_peaks_df` (época em que cada norma de gradiente atinge o pico);
a matriz de confusão é impressa em números antes da figura, e a última célula fecha com o `history_df.tail()`
(épocas 26–30). O notebook mostra as tabelas com 4 casas decimais (`pd.set_option('display.precision', 4)`),
então os números abaixo são exatamente os que aparecem na tela com `SEED = 42`:

| modelo | accuracy | macro_f1 | log_loss |
|---|---|---|---|
| Regressão logística | 0.9556 | 0.9555 | 0.1894 |
| MLP com backpropagation | **0.9583** | **0.9584** | **0.1429** |

Números-chave para levar da aula:

- O ganho de acurácia do MLP é de apenas **+0.28 ponto percentual** (0.9556 → 0.9583). Sozinho, isso
  não sustentaria uma tese — o baseline linear é mesmo muito forte no `digits`.
- O ganho **real** aparece na `log loss`: **0.1894 → 0.1429**, uma queda de ~25%. O MLP não acerta
  muito mais; ele acerta com **muito mais calibração e confiança**. É um bom gancho para discutir por
  que acurácia não é a métrica certa quando o modelo alimenta uma decisão de crédito ou de churn.
- A curva de treino (`history_df`, 30 épocas) sai de `val_accuracy` **0.4039 na época 1** (primeira linha
  do `history_df.head()`) e chega a **0.9582 na época 30** (última linha do `history_df.tail()`), com
  `val_loss` caindo de 2.2332 para **0.1364**. A maior `val_accuracy` foi na época 28 (0.9610), mas o
  critério de seleção do `best_state` é a `val_loss`, e por ela a época 30 vence.
- As normas de gradiente confirmam a hipótese: a primeira camada começa em **0.0966**, sobe até **0.3113
  na época 9** e termina em **0.2132**; a última camada vai de 0.0943 a um pico de **0.6522 na época 10**
  e desce para **0.3122** (os picos estão no `grad_peaks_df`). Nenhum colapso para zero — o gradiente
  **chega** à primeira camada, e é por isso que o treino converge.
- A matriz de confusão impressa mostra a diagonal carregada e poucas confusões: o maior bloco fora da
  diagonal é o dígito 2 lido como 1 (4 casos); seguem 8 lido como 1 (2 casos) e 9 lido como 7 (2 casos).
- O balanceamento das classes é quase perfeito (104 a 110 amostras de treino por dígito), então macro-F1
  e acurácia andam juntos — o que **não** aconteceria num dataset desbalanceado.

Figuras esperadas: grade de amostras do `digits`, curvas de perda/acurácia por época com as normas de
gradiente, e o painel de erros residuais com os dígitos que o modelo confundiu.
