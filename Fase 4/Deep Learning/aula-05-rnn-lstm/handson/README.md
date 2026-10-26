# Hands-on — Aula 05: RNN, GRU e LSTM em Forecasting Temporal

Caso de negócio de referência: **iFood — previsão de ETA e demanda**.

## O que este notebook investiga

A pergunta é **quando** a memória explícita de arquiteturas recorrentes realmente agrega valor em
séries temporais.

O notebook gera uma série sintética de demanda com **1.826 dias** (2021-01-01 a 2025-12-31) cujo
mecanismo gerador é **deliberadamente não-linear e com estado latente de longo prazo**:

1. **Saturação por estoque** — `demanda_observada = min(demanda_latente, estoque)`. O estoque cai a cada
   venda e só é reposto às segundas. Em **11,7% dos dias** a prateleira esvazia e a série satura.
2. **Regime latente de campanha** — quando a soma dos últimos 14 dias cruza um limiar, uma campanha liga
   por 12 dias e adiciona um patamar. O gatilho é um **acumulado**, não o último valor. Fica ativa em
   **54,8% dos dias**.
3. **Fadiga de promoção** — o ganho da promoção é multiplicado por `1 - exp(-dias_desde_a_última / 8)`.
   O efeito depende da **ordem e do espaçamento** dos eventos, não do valor defasado.
4. **Interação multiplicativa** — o calor só conta acima de 24 °C e pesa 2,1× mais no fim de semana.

Estoque e estado da campanha ficam **latentes**: as covariáveis visíveis são só `demanda`, `promo`,
`temperatura` e `fim de semana`.

Seis modelos disputam a previsão de curto prazo, sobre uma **divisão cronológica** treino/validação/teste
(1.258 / 270 / 270 janelas, sem vazamento temporal):

1. **Persistência** — "amanhã é igual a hoje";
2. **Naive sazonal** — "amanhã é igual ao mesmo dia da semana passada" (a referência do MASE);
3. **Ridge autoregressivo** — baseline linear sobre a janela de 28 dias × 4 features (112 colunas);
4. **RNN vanilla**;
5. **GRU**;
6. **LSTM**.

A avaliação usa **MAE, RMSE, sMAPE e MASE**, com análise do erro **em dias de promoção** e **em dias de
campanha ativa** contra dias normais.

> **Por que o gerador mudou.** A versão anterior deste notebook gerava a série com
> `demand[t] = 20 + 0.5·lag_1 + 0.15·lag_7 + ...`, ou seja, um processo **linear**. Nesse desenho o Ridge
> com defasagens era o estimador **ótimo por construção** e a LSTM não tinha o que aprender (MASE 1.76,
> pior que a persistência). O aluno saía com a conclusão falsa de que "LSTM não serve para série temporal",
> quando o experimento só media um artefato do gerador. O protocolo (split cronológico, janela, MASE,
> comparação com persistência e Ridge) foi mantido; o que mudou foi a **dinâmica dos dados**.

## Pergunta experimental

> Quando a dinâmica da série é não-linear (saturação por estoque, limiar de campanha, interação
> temperatura × fim de semana) e depende de um estado latente acumulado, a memória de GRU e LSTM melhora
> de forma consistente o forecasting de curto prazo em relação a baselines lineares?

**Hipótese do notebook:** a persistência vai mal por causa da sazonalidade semanal; o Ridge é forte mas
esbarra no teto do que uma função linear das defasagens representa; a RNN vanilla fica no meio; GRU e LSTM
ficam **abaixo de MASE 1 e abaixo do Ridge**.

> **O experimento confirma a hipótese.** GRU e LSTM ficam em MASE 0,68 e 0,67, contra 0,85 do Ridge — e
> a vantagem se concentra exatamente nos regimes que exigem acumulado latente.

## Como executar

1. Abra `hands-on.ipynb` no VS Code ou no Jupyter.
2. Selecione o kernel do ambiente virtual do curso: `C:\git\aula\.venv`
   (aparece como **"Python (curso IA Generativa)"** / **".venv"**). Sem esse kernel, o `torch` não é encontrado.
3. Execute **Run All**, sempre do topo. Aqui a ordem é ainda mais crítica que nas outras aulas: a
   divisão cronológica e o cálculo do MASE dependem das células anteriores terem definido a escala
   ingênua da série.

Alternativa por linha de comando:

```bash
PYTHONUTF8=1 JUPYTER_PATH="C:/git/aula/.venv/share/jupyter" \
  C:/git/aula/.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace hands-on.ipynb
```

A série é gerada dentro do notebook — nenhuma fonte externa de dados é usada.

## Tempo aproximado

**~50-70 segundos** de ponta a ponta em GPU (60 épocas para cada uma das três arquiteturas recorrentes).
Em CPU, conte cerca de 3 minutos.

## Saídas esperadas

Ranking final (`results_df`, ordenado por MAE) — escala ingênua sazonal do treino: **6.582**:

| modelo | MAE | RMSE | sMAPE | MASE | parâmetros |
|---|---|---|---|---|---|
| **LSTM** | **4.42** | 6.71 | 0.1596 | **0.672** | 10.417 |
| **GRU** | 4.49 | 6.96 | 0.1298 | **0.682** | 7.825 |
| RNN | 5.29 | 7.68 | 0.1676 | 0.805 | 2.641 |
| Ridge autoregressivo | 5.60 | 8.42 | 0.1713 | 0.851 | 113 |
| Naive sazonal | 6.53 | 10.91 | 0.1374 | 0.991 | 0 |
| Persistência | 24.43 | 35.54 | 0.5696 | 3.712 | 0 |

Números-chave para levar da aula:

- **As células com gates vencem, e vencem por um mecanismo identificável.** LSTM (MASE **0.672**) e GRU
  (**0.682**) erram cerca de **21% menos** que o Ridge (**0.851**) e ficam confortavelmente abaixo de 1,
  isto é, batem o naive sazonal.
- **O Ridge não é um espantalho.** Com MASE **0.851** ele bate a persistência e o naive sazonal usando
  apenas **113 parâmetros**. A lição não é "linear é ruim", é "linear tem um teto quando a dinâmica é
  não-linear".
- **A RNN vanilla fica no meio (MASE 0.805)** — melhor que o Ridge, pior que GRU/LSTM. É a evidência de
  que o ganho vem de *memória com gates*, não de "ser uma rede neural".
- **Persistência é inviável (MASE 3.712, MAE 24.43)** por causa da sazonalidade semanal forte: repetir
  ontem é péssimo quando sábado não se parece com sexta.
- **O erro por regime** (`promo_mae`, sobre 270 dias de teste — 22 com promoção, 264 em campanha) mostra
  onde a memória paga:

  | modelo | MAE sem promoção | MAE com promoção | MAE fora de campanha | MAE em campanha |
  |---|---|---|---|---|
  | persistência | 24.16 | 27.48 | 18.36 | 24.57 |
  | ridge | 4.72 | 15.53 | 12.77 | 5.44 |
  | rnn | 4.31 | 16.38 | 9.66 | 5.20 |
  | gru | 3.67 | **13.70** | 12.53 | **4.31** |
  | lstm | **3.60** | **13.67** | 11.65 | **4.26** |

  Em dias de promoção a LSTM erra **13.67** contra **15.53** do Ridge; em dias de campanha ativa,
  **4.26** contra **5.44**. É exatamente nos regimes que dependem de acumulado latente que a diferença
  aparece.
- **Os 10 maiores erros da GRU** ainda se concentram em dias de promoção (**9 dos 10** têm `promo = 1`),
  com erro absoluto de até **30.6** unidades — mas a tabela mostra o Ridge errando o mesmo ou mais nos
  mesmos dias (27.5, 32.1, 32.3...). Promoção continua sendo o regime mais difícil para todo mundo;
  a diferença é que agora existe um modelo que degrada menos.

Figuras esperadas: a série completa com marcação das promoções e da campanha latente, um zoom de 120 dias
destacando os **dias saturados pelo estoque**, curvas de loss de validação por arquitetura e o gráfico de
previsão contra valor real nos últimos 120 pontos do teste.
