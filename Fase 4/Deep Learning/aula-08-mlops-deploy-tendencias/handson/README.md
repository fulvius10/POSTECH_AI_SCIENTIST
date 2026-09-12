# Hands-on — Aula 08: Tracking, Export, Quantização e Drift

Caso de negócio de referência: **Americanas — busca multimodal e monitoramento operacional**.

## O que este notebook investiga

Nesta prática final o modelo deixa de ser um resultado acadêmico e vira **artefato operacional**. O
fluxo vai da seleção do candidato até uma checagem mínima de prontidão para produção:

1. **Ledger de runs** — dois candidatos (`mlp_small` e `mlp_large`) treinados e registrados com
   parâmetros, métricas e caminho dos artefatos, num `DataFrame` local reproduzível (o papel que o
   MLflow cumpriria);
2. **Export e quantização** — o melhor candidato é exportado para **ONNX** e também quantizado
   dinamicamente para **int8**;
3. **Benchmark operacional** — acurácia, log loss, **latência p50/p95** e **tamanho do artefato** das
   três variantes (FP32 PyTorch, int8 dinâmico, ONNX Runtime);
4. **Monitor de drift** — um conjunto deslocado por ruído aditivo e aumento de brilho, com **PSI**
   (Population Stability Index) sobre confiança e intensidade de pixel, e um **gate de release** com
   quatro critérios objetivos.

## Pergunta experimental

> Depois de escolher um modelo, até que ponto export, quantização e drift alteram o comportamento
> observado offline?

**Hipótese do notebook:** dois candidatos com capacidades ligeiramente diferentes mostram trade-off
qualidade/custo; a quantização dinâmica reduz tamanho e latência com perda pequena; e um conjunto
deslocado eleva o PSI e derruba o desempenho, justificando alerta.

## Como executar

1. Abra `hands-on.ipynb` no VS Code ou no Jupyter.
2. Selecione o kernel do ambiente virtual do curso: `C:\git\aula\.venv`
   (aparece como **"Python (curso IA Generativa)"** / **".venv"**). Sem esse kernel, o `torch` não é encontrado.
3. Execute **Run All**. O notebook cria a pasta `artifacts_mlops/` ao lado dele, com os pesos `.pt`,
   o modelo `.onnx` e a versão int8.

Alternativa por linha de comando:

```bash
PYTHONUTF8=1 JUPYTER_PATH="C:/git/aula/.venv/share/jupyter" \
  C:/git/aula/.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace hands-on.ipynb
```

### Dependências opcionais do caminho ONNX

O export ONNX precisa de **três** pacotes: `onnxruntime` (para executar), `onnx` e `onnxscript` (para
exportar). Se algum faltar, o notebook **não quebra**: ele captura o `ModuleNotFoundError`, deixa a
linha `onnx_export` com `NaN` e registra a limitação na coluna `onnx_export_error`. Para ter o
comparativo completo:

```bash
C:/git/aula/.venv/Scripts/python.exe -m pip install onnx onnxscript
```

## Tempo aproximado

**~15 a 20 segundos** de ponta a ponta (20 épocas para cada um dos dois candidatos, mais export,
quantização e benchmark).

## Saídas esperadas

Seleção do candidato (`candidate_df`, ordenada por `val_log_loss`):

| candidato | val_accuracy | val_log_loss | test_accuracy |
|---|---|---|---|
| `mlp_large` (128, 64) | **0.9415** | **0.2088** | **0.9472** |
| `mlp_small` (64, 32) | 0.8914 | 0.3725 | 0.8750 |

Benchmark operacional (`operational_df`):

| variante | accuracy | log_loss | p50 (ms) | p95 (ms) | tamanho (KB) |
|---|---|---|---|---|---|
| `fp32_pytorch` | 0.9472 | 0.2333 | 0.39 | 0.56 | **70.4** |
| `int8_dynamic` | 0.9472 | 0.2338 | 0.40 | 0.57 | **22.5** |
| `onnx_export` | — | — | **0.09** | **0.21** | 1.6 (grafo) |

Métricas sob drift (impressas no início da última célula de código, antes de `monitor_df`):

```
Nominal: accuracy=0.9472 | log_loss=0.2333
Drift:   accuracy=0.9056 | log_loss=0.2748 | queda de acurácia=0.0417
```

Monitor de drift (`monitor_df`): `confidence_max` com PSI **0.041**; `pixel_intensity_mean` com PSI **6.98**.
Gate de release (`gate_df`): os quatro critérios com `status = True`.

Números-chave para levar da aula:

- **A quantização int8 é praticamente de graça.** Acurácia **idêntica** (0.9472 nas duas variantes) e
  log loss piorando na **quarta casa decimal** (0.23329 → 0.23377), em troca de **1/3 do tamanho**
  (70.4 KB → 22.5 KB). A latência p50 do int8 (0.40 ms, na CPU) ficou **empatada** com a do FP32
  (0.39 ms, na GPU): para uma rede desta escala, o ganho da quantização é de tamanho, não de tempo.
- **O ONNX Runtime é o maior ganho de latência:** p50 de **0.09 ms** contra 0.39 ms do PyTorch FP32,
  algo em torno de **4x mais rápido** para este modelo pequeno. Atenção à comparação: o FP32 foi medido
  na GPU (`device = 'cuda'`), enquanto int8 e ONNX Runtime rodaram na CPU — para uma rede desse
  tamanho, o custo de despachar para a placa domina a conta, e a GPU não sai na frente.
- **Cuidado ao ler o "1.6 KB" do ONNX:** com o exportador novo do PyTorch, os pesos vão para um arquivo
  externo `mlp_large.onnx.data` (~67 KB). O artefato real é a soma dos dois — o notebook mede só o
  grafo. É exatamente o tipo de armadilha que aparece quando se empacota modelo para serving.
- **O drift derruba o modelo de forma mensurável:** a acurácia cai de **0.9472 (nominal) para 0.9056
  (com drift)**, uma perda de **4.2 pontos percentuais**, e a log loss sobe de 0.2333 para 0.2748.
- **PSI mostra onde o drift está.** O sinal `confidence_max` tem PSI de apenas **0.041** (estável,
  abaixo do limiar clássico de 0.2), mas `pixel_intensity_mean` explode para **6.98**. Ou seja: a
  distribuição de **entrada** mudou brutalmente e a confiança do modelo mal percebeu. Monitorar só a
  saída do modelo teria deixado o problema passar.
- **O gate de release passa nos 4 critérios** (`accuracy_nominal >= 0.94`, `quantized_drop <= 0.01`,
  `confidence_psi < 0.2`, `drift_accuracy_drop <= 0.08`) — vale discutir em sala se `drift_accuracy_drop
  <= 0.08` é uma tolerância aceitável para o caso de negócio, dado que a queda observada foi de 0.042.

> **Nota:** as latências (`p50_ms`, `p95_ms`) variam entre execuções conforme a carga da máquina —
> espere oscilação de dezenas de por cento, e int8 e FP32 podem trocar de posição entre uma execução
> e outra. O que é estável: o ONNX Runtime é sempre o mais rápido, e todos os números de qualidade
> (acurácia, log loss, PSI, tamanhos).

Figuras esperadas: grade 2×5 de amostras nominais contra amostras com drift, e o painel com o
histograma da confiança máxima (nominal × drift) ao lado da matriz de confusão sob drift.

> **Nota:** as tabelas intermediárias são exibidas com `display(...)` dentro das próprias células: a
> célula de treino mostra `candidate_df` (acima dos avisos do exportador) e termina em `operational_df`;
> a célula de drift mostra as métricas nominal × drift, `monitor_df` e `gate_df`, e termina em
> `tracker.to_frame()` seguido das figuras.
