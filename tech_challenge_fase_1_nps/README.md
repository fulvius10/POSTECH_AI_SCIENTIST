# Tech Challenge Fase 1 — Case NPS Preditivo

Projeto desenvolvido para o Tech Challenge da Fase 1 da Pós-Tech AI Scientist.

O objetivo é analisar a experiência dos clientes de um e-commerce, identificar sinais operacionais associados à insatisfação e demonstrar como esses sinais podem ser utilizados para antecipar clientes com maior risco de se tornarem detratores antes da aplicação da pesquisa de NPS.

## Problema de negócio

Atualmente, a empresa identifica a insatisfação apenas após o encerramento da jornada, quando o cliente responde à pesquisa de NPS.

O desafio é transformar informações já existentes na operação — como entrega, reclamações e atendimento — em insights que permitam priorizar ações de recuperação antes da aplicação da pesquisa.

A pergunta principal é:

> Quais sinais operacionais estão associados à insatisfação e podem ajudar a antecipar clientes com maior risco de se tornarem detratores?

Os resultados podem apoiar áreas como logística, atendimento, CRM, produto e experiência do cliente.

## Base de dados

A base utilizada contém:

- 2.500 pedidos;
- 19 variáveis originais;
- nenhum valor ausente;
- nenhuma linha duplicada;
- `customer_id` único;
- `order_id` único.

Cada registro representa um pedido associado a um cliente.

As variáveis abrangem:

- características do cliente;
- informações do pedido;
- logística;
- atendimento;
- reclamações;
- indicadores de satisfação.

## Definição do NPS

A variável original de satisfação é:

`nps_score`

Ela varia de 0 a 10.

Foram adotadas as seguintes categorias:

- **Detrator:** `nps_score <= 6`;
- **Neutro:** `6 < nps_score < 9`;
- **Promotor:** `nps_score >= 9`.

A base possui notas decimais, por isso os intervalos foram tratados dessa forma.

O NPS tradicional é calculado como:

`% Promotores - % Detratores`

Na amostra:

- detratores: 74,0%;
- neutros: 21,6%;
- promotores: 4,4%;
- NPS tradicional: **-69,6**;
- média da nota de NPS: **4,379**.

A média da nota e o NPS tradicional são métricas diferentes.

## Análise exploratória

A EDA foi orientada a perguntas de negócio e buscou identificar possíveis pontos de ruptura na experiência.

### Atraso da entrega

| Faixa | Pedidos | NPS médio | Detratores |
|---|---:|---:|---:|
| Sem atraso | 277 | 6,86 | 36,5% |
| 1-2 dias | 1.261 | 5,05 | 67,7% |
| 3-4 dias | 795 | 3,10 | 91,8% |
| 5+ dias | 167 | 1,28 | 99,4% |

A partir da faixa de três dias de atraso, mais de 90% dos pedidos são detratores nesta amostra.

### Reclamações

| Faixa | Pedidos | NPS médio | Detratores |
|---|---:|---:|---:|
| 0-1 | 145 | 7,89 | 7,6% |
| 2-3 | 784 | 5,31 | 59,1% |
| 4-5 | 1.044 | 3,98 | 84,2% |
| 6+ | 527 | 2,82 | 94,5% |

### Contatos com atendimento

A taxa de detratores aumenta conforme crescem os contatos:

- 0 contatos: 59,2%;
- 1 contato: 70,3%;
- 2 contatos: 79,1%;
- 3 ou mais: 90,4%.

### Tempo de resolução

Também foi observado crescimento da taxa de detratores conforme aumenta o tempo de resolução:

- 0-2 dias: 64,2%;
- 3-5 dias: 73,2%;
- 6-8 dias: 77,4%;
- 9+ dias: 81,5%.

Esses resultados mostram **associações**, e não evidência de causalidade.

## Estratégia preditiva

Como etapa adicional do Tech Challenge, foi implementado um modelo de classificação.

A variável-alvo utilizada foi:

`is_detractor`

Definição:

- `1`: `nps_score <= 6`;
- `0`: `nps_score > 6`.

A classificação foi escolhida porque o uso operacional desejado é priorizar clientes com maior risco de detração.

## Prevenção de data leakage

As seguintes variáveis foram excluídas do treinamento:

- `customer_id`;
- `order_id`;
- `nps_score`;
- `nps_category`;
- `is_detractor`;
- `repeat_purchase_30d`;
- `csat_internal_score`.

`repeat_purchase_30d` contém informação posterior ao pedido.

O momento de coleta de `csat_internal_score` não está documentado, portanto foi excluído de forma conservadora.

O modelo considera como momento de scoring o período após a consolidação dos dados operacionais de entrega e atendimento, porém antes da coleta do NPS.

## Preparação para Machine Learning

Foi utilizado um `Pipeline` do scikit-learn.

Variáveis numéricas:

- `StandardScaler`.

Variáveis categóricas:

- `OneHotEncoder(handle_unknown="ignore")`.

Os dados foram separados em:

- 2.000 registros para treinamento;
- 500 registros para holdout final.

A separação foi estratificada para preservar a proporção de detratores.

## Seleção do modelo

Foram comparados:

1. `DummyClassifier`;
2. Regressão Logística;
3. Random Forest.

A seleção foi realizada somente no conjunto de treinamento através de **Stratified 5-Fold Cross-Validation**.

Resultados médios:

| Modelo | ROC-AUC CV | Average Precision CV |
|---|---:|---:|
| Regressão Logística | **0,875** | **0,951** |
| Random Forest | 0,870 | 0,949 |
| Baseline | 0,500 | 0,741 |

A regressão logística foi escolhida pela maior ROC-AUC média na validação cruzada.

Também apresentou maior estabilidade entre treino e validação:

- ROC-AUC médio no treino: aproximadamente 0,886;
- ROC-AUC médio na validação: aproximadamente 0,875.

O Random Forest apresentou ROC-AUC de aproximadamente 0,985 no treino e 0,870 na validação, indicando maior tendência a overfitting nesta configuração.

## Avaliação no holdout

Após a seleção, a regressão logística foi avaliada uma única vez no conjunto de 500 registros de holdout.

Resultados:

- ROC-AUC: **0,877**;
- Average Precision: **0,947**;
- acurácia: **0,788**;
- precisão para detratores: **0,915**;
- recall para detratores: **0,786**;
- F1 para detratores: **0,846**.

Matriz de confusão:

| | Previsto não detrator | Previsto detrator |
|---|---:|---:|
| Não detrator | 103 | 27 |
| Detrator | 79 | 291 |

O modelo identificou 291 dos 370 detratores existentes no holdout.

O baseline apresenta recall igual a 1,0 porque classifica todos os clientes como detratores. Sua ROC-AUC de 0,5 mostra que ele não possui capacidade real de ordenação de risco.

## Importância das variáveis

Foi utilizada importância por permutação baseada em ROC-AUC.

As três variáveis com maior contribuição incremental foram:

1. `complaints_count`;
2. `delivery_delay_days`;
3. `resolution_time_days`.

A importância por permutação mede quanto o desempenho do modelo piora quando uma variável é embaralhada, mantendo as demais disponíveis.

Portanto, não deve ser interpretada como causalidade.

## Uso operacional

Após o treinamento, o pipeline escolhido é salvo em:

`models/detractor_classifier.joblib`

O script:

`src/score_new_orders.py`

carrega o modelo e gera:

- `detractor_probability`;
- `risk_band`.

Faixas iniciais utilizadas:

- **Baixo:** probabilidade até 0,50;
- **Alto:** acima de 0,50 até 0,75;
- **Crítico:** acima de 0,75.

Os pedidos são ordenados da maior para a menor pontuação, simulando uma fila operacional de recuperação.

Esses limites são apenas uma proposta inicial e devem ser calibrados de acordo com custos e capacidade operacional.

## Estrutura do projeto

```text
tech_challenge_fase_1_nps/
├── data/
│   ├── raw/
│   │   └── desafio_nps_fase_1.csv
│   └── processed/
│       ├── nps_prepared.csv
│       ├── test_predictions.csv
│       └── scored_orders.csv
├── models/
│   └── detractor_classifier.joblib
├── reports/
│   ├── figures/
│   ├── business_summary.json
│   ├── data_quality.csv
│   ├── model_cv_metrics.csv
│   ├── model_details.json
│   ├── model_feature_importance.csv
│   ├── model_metrics.csv
│   └── segment_*.csv
├── src/
│   ├── run_analysis.py
│   └── score_new_orders.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Como executar

Criar o ambiente:

```powershell
python -m venv .venv
```

Ativar:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar dependências:

```powershell
python -m pip install -r requirements.txt
```

Executar análise e treinamento:

```powershell
python src\run_analysis.py
```

Pontuar pedidos:

```powershell
python src\score_new_orders.py --input data\raw\desafio_nps_fase_1.csv
```

## Principais arquivos gerados

### `data/processed/nps_prepared.csv`

Base preparada com variáveis derivadas para análise.

### `data/processed/test_predictions.csv`

Previsões realizadas no holdout utilizado para avaliação final.

### `data/processed/scored_orders.csv`

Demonstração do processo de scoring e priorização operacional.

### `reports/model_cv_metrics.csv`

Resultados da validação cruzada utilizada para seleção do modelo.

### `reports/model_metrics.csv`

Avaliação dos modelos no holdout.

### `reports/model_feature_importance.csv`

Importância das variáveis por permutação.

### `models/detractor_classifier.joblib`

Pipeline treinado utilizado para inferência.

## Limitações

- a base não contém datas, impossibilitando validação temporal;
- existe apenas um pedido por cliente;
- não há informação sobre SLA prometido;
- não há transportadora ou categoria do produto;
- não existe informação sobre taxa de resposta à pesquisa;
- `csat_internal_score` possui momento de coleta não documentado;
- as notas de NPS possuem valores decimais;
- não existe experimento controlado que permita inferência causal;
- as probabilidades não foram submetidas a uma etapa específica de calibração;
- os limites das faixas de risco ainda precisam ser validados operacionalmente;
- o modelo ainda não foi validado prospectivamente em produção.

## Conclusão

A análise indica que o perfil de maior risco nesta amostra é principalmente operacional.

Reclamações acumuladas, atrasos na entrega e maior tempo de resolução aparecem associados à insatisfação e também apresentam relevância para o modelo preditivo.

A solução proposta transforma esses sinais em uma pontuação que pode ser utilizada para priorizar ações de recuperação antes da aplicação da pesquisa de NPS.

Os resultados devem apoiar investigação e tomada de decisão, e não ser interpretados como evidência causal.