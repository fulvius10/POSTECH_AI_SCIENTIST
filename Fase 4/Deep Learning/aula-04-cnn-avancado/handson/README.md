# Hands-on — Aula 04: ResNet Pequena, Transfer Learning e Grad-CAM

Caso de negócio de referência: **Magalu — busca visual**.

## O que este notebook investiga

Esta prática simula um cenário muito comum fora dos grandes laboratórios: **poucos rótulos no domínio-alvo**
e necessidade de reaproveitar representação aprendida antes.

Em vez de baixar ImageNet, o notebook gera dois datasets sintéticos de formas geométricas (32x32):

- **domínio-fonte** — formas limpas, centralizadas, bem contrastadas (usado para pré-treinar o backbone residual);
- **domínio-alvo** — as mesmas classes, mas com ruído, rotação, deslocamento, mudança de contraste e
  **muito menos exemplos por classe**.

Três regimes de treino são comparados no domínio-alvo:

1. **Treino do zero** (`scratch`) — baseline honesto, sem transferência;
2. **Backbone congelado** (`frozen_transfer`) — *feature extraction*: só a cabeça de classificação treina;
3. **Fine-tuning completo** (`full_finetune`) — todo o backbone é destravado.

Ao final, **Grad-CAM** mostra onde a rede concentrou evidência visual — útil para flagrar quando o
modelo está se apoiando no ruído de fundo em vez da forma.

## Pergunta experimental

> Quando o conjunto-alvo é pequeno e visualmente mais ruidoso do que o domínio-fonte, vale a pena
> reutilizar um backbone pré-treinado?

**Hipótese do notebook:** treino do zero sofre com variância e baixa eficiência amostral; o backbone
pré-treinado oferece ponto de partida melhor, tanto congelado quanto em fine-tuning leve.

## Como executar

1. Abra `hands-on.ipynb` no VS Code ou no Jupyter.
2. Selecione o kernel do ambiente virtual do curso: `C:\git\aula\.venv`
   (aparece como **"Python (curso IA Generativa)"** / **".venv"**). Sem esse kernel, o `torch` não é encontrado.
3. Execute **Run All**. Os três regimes precisam rodar em ordem: o `frozen_transfer` e o `full_finetune`
   consomem os pesos produzidos pelo pré-treino no domínio-fonte.

Alternativa por linha de comando:

```bash
PYTHONUTF8=1 JUPYTER_PATH="C:/git/aula/.venv/share/jupyter" \
  C:/git/aula/.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace hands-on.ipynb
```

Nenhum download de imagem é necessário — as formas são desenhadas dentro do notebook. Roda em CPU;
usa GPU se disponível.

## Tempo aproximado

**~15 segundos** de ponta a ponta (10 épocas no domínio-fonte + 12 épocas em cada um dos três regimes).

## Saídas esperadas

Comparação final (`results_df`, exibida na célula de treino logo acima das curvas e repetida ao fim da célula do Grad-CAM):

| regime | accuracy | macro_f1 | balanced_accuracy |
|---|---|---|---|
| Treino do zero | **0.2571** | 0.1023 | 0.2500 |
| Backbone congelado | **0.7000** | 0.6974 | 0.6961 |
| Fine-tuning completo | **0.7429** | 0.7393 | 0.7435 |

Números-chave para levar da aula:

- **Treino do zero simplesmente não sai do lugar:** 0.2571 de acurácia com 4 classes é praticamente o
  chute (0.25). A tabela `scratch_history`, exibida na mesma célula logo abaixo de `results_df`, mostra a
  `train_accuracy` **cravada em 0.25 nas 12 épocas** e a `train_loss` *subindo* de 1.3964 (≈ ln 4, o chute
  uniforme) para picos acima de 2.4 (2.4339 na época 10). Com esse volume de rótulos, a rede não tem de onde aprender.
- **Transferir resolve o problema:** congelar o backbone leva a acurácia de 0.2571 para **0.7000** (49 acertos
  em 70, como a célula do Grad-CAM imprime) — um salto de **+44 pontos percentuais** sem treinar uma única
  camada convolucional.
- **Fine-tuning completo fica à frente, por pouco:** **0.7429** de acurácia (52 acertos em 70), 0.7393 de macro-F1
  e 0.7435 de balanced accuracy — **3 imagens acima do congelado**. Com 70 imagens de teste cada imagem vale 1,4 pp,
  então a diferença de 4,3 pp é sugestiva, não conclusiva. Vale conversar sobre custo/benefício: destravar o
  backbone inteiro custa mais computação e, com 140 imagens de treino, o ganho é pequeno.
- **A curva de validação do fine-tuning mostra o risco** (linha vermelha, gráfico da esquerda): parte acima da
  do congelado, despenca ao nível do chute entre as épocas 4 e 9 — *catastrophic forgetting* em miniatura — e só
  recupera nas três últimas épocas; a perda de validação (gráfico da direita) faz um pico na época 6 e termina
  abaixo da do congelado. Como `train_model` guarda o melhor estado pela perda de validação, o modelo que vai ao
  teste é o das últimas épocas, depois da recuperação.
- O `macro_f1` do treino do zero (**0.1023**) denuncia o colapso: o modelo está prevendo essencialmente
  uma classe só — a matriz de confusão do treino do zero, impressa na célula do Grad-CAM, mostra as 70 imagens
  de teste previstas como `circulo` (primeira coluna: 18 / 17 / 18 / 17). Compare com 0.6974 e 0.7393 dos
  regimes transferidos.
- **Matriz de confusão do fine-tuning** (impressa na célula do Grad-CAM; ordem círculo, quadrado, triângulo, cruz):
  diagonal **10 / 9 / 16 / 17**. Triângulo e cruz praticamente resolvidos; o erro se concentra em círculo→quadrado (6)
  e quadrado→cruz (8) — a coluna `cruz` atrai 12 dos 18 erros.
- O pré-treino no domínio-fonte atinge **100% de acurácia de validação já na época 5** (tabela `source_history`,
  exibida na célula de treino acima do gráfico: `val_accuracy` = 1.0 da época 5 em diante; e a curva "pré-treino
  fonte" no gráfico da esquerda) — ou seja,
  o domínio-fonte é fácil de propósito, e é exatamente essa representação "barata" que transfere.

> **Nota sobre reprodutibilidade:** a célula de setup fixa a semente (42) e liga as flags de determinismo do cuDNN
> (`cudnn.deterministic = True`, `cudnn.benchmark = False`, `use_deterministic_algorithms(True)` e
> `CUBLAS_WORKSPACE_CONFIG`). Com isso a execução é **reprodutível nesta GPU** — duas execuções seguidas devolvem
> exatamente as mesmas tabelas, matrizes e curvas. Em CPU ou em outra GPU os números podem variar alguns pontos,
> sobretudo no fine-tuning completo. O resultado que importa se mantém: **transferir ≫ treinar do zero**; a
> vantagem de 3 imagens do fine-tuning sobre o congelado é a parte a tratar com ceticismo.

Saídas em texto: `results_df`, `source_history` e `scratch_history` (célula de treino, acima das curvas) e as
matrizes de confusão numéricas com a contagem de acertos (célula do Grad-CAM, acima das figuras).

Figuras esperadas: grades de preview dos domínios fonte e alvo, curvas de treino dos três regimes,
matriz de confusão e o painel **Grad-CAM** com os mapas de calor sobre as formas.
