# Análise de negócio — Case NPS Preditivo

## 1. Problema de negócio

O e-commerce identifica a satisfação somente após o encerramento da jornada, quando o cliente responde à pesquisa de NPS.

Isso limita a capacidade da empresa de antecipar problemas e atuar enquanto ainda existe oportunidade de recuperação da experiência.

O objetivo do projeto é transformar sinais operacionais já disponíveis em informações que ajudem a identificar clientes em maior risco de insatisfação.

## 2. Stakeholders

### Logística

Pode utilizar os resultados para investigar atrasos, tentativas de entrega e possíveis gargalos operacionais.

### Atendimento

Pode identificar casos com reclamações acumuladas, contatos recorrentes e altos tempos de resolução.

### CRM e Marketing

Pode diferenciar clientes que precisam de recuperação daqueles mais adequados a ações de fidelização.

### Produto e jornada

Pode investigar fricções recorrentes identificadas durante o atendimento.

### Experiência do Cliente e estratégia

Pode acompanhar indicadores de satisfação e coordenar ações entre diferentes áreas da empresa.

## 3. Importância do NPS

O NPS funciona como um indicador da disposição do cliente em recomendar a empresa.

No e-commerce, uma experiência negativa pode estar associada a menor recompra, comunicação negativa e perda de relacionamento.

O NPS, porém, não substitui métricas financeiras ou operacionais.

Indicadores complementares relevantes incluem:

- recompra;
- churn;
- lifetime value;
- SLA logístico;
- entrega no prazo;
- devoluções;
- cancelamentos;
- CSAT;
- CES;
- first contact resolution;
- custo por atendimento;
- taxa de resposta da pesquisa;
- reclamações públicas;
- conversão;
- market share.

## 4. Target

A variável original de satisfação é `nps_score`.

A partir dela foi criada:

`is_detractor`

Regra:

- `1`: NPS até 6;
- `0`: NPS acima de 6.

A abordagem de classificação foi escolhida porque a decisão de negócio desejada é identificar e priorizar clientes com maior risco de detração.

## 5. Diagnóstico geral

A base possui:

- 2.500 pedidos;
- 19 variáveis originais;
- zero valores ausentes;
- zero duplicidades.

Distribuição:

- 74,0% detratores;
- 21,6% neutros;
- 4,4% promotores.

NPS tradicional:

**-69,6**

Média da nota de NPS:

**4,379**

## 6. Atraso

Sem atraso:

- 277 pedidos;
- NPS médio 6,86;
- 36,5% detratores.

1-2 dias:

- 1.261 pedidos;
- NPS médio 5,05;
- 67,7% detratores.

3-4 dias:

- 795 pedidos;
- NPS médio 3,10;
- 91,8% detratores.

5 ou mais dias:

- 167 pedidos;
- NPS médio 1,28;
- 99,4% detratores.

A faixa iniciada em três dias representa um patamar operacional crítico nesta amostra.

## 7. Reclamações

0-1 reclamação:

- NPS médio 7,89;
- 7,6% detratores.

2-3 reclamações:

- NPS médio 5,31;
- 59,1% detratores.

4-5 reclamações:

- NPS médio 3,98;
- 84,2% detratores.

6 ou mais:

- NPS médio 2,82;
- 94,5% detratores.

Reclamações acumuladas apresentam uma das associações mais fortes com insatisfação observadas na análise.

## 8. Contatos

A taxa de detratores aumenta progressivamente:

- nenhum contato: 59,2%;
- um contato: 70,3%;
- dois contatos: 79,1%;
- três ou mais: 90,4%.

Isso não significa que entrar em contato cause insatisfação.

Contatos repetidos podem ser consequência de um problema anterior que não foi resolvido adequadamente.

## 9. Tempo de resolução

Taxa de detratores:

- 0-2 dias: 64,2%;
- 3-5 dias: 73,2%;
- 6-8 dias: 77,4%;
- 9+ dias: 81,5%.

Quanto maior o tempo de resolução, maior a proporção de detratores observada na amostra.

## 10. Modelo preditivo

Foram comparados:

- baseline;
- regressão logística;
- Random Forest.

A escolha foi realizada por Stratified 5-Fold Cross-Validation no conjunto de treinamento.

A regressão logística apresentou:

- ROC-AUC CV médio: 0,875;
- desvio padrão: 0,012;
- Average Precision CV: 0,951.

A regressão logística também apresentou pequena diferença entre ROC-AUC de treino e validação.

O Random Forest apresentou aproximadamente:

- ROC-AUC treino: 0,985;
- ROC-AUC CV: 0,870.

Essa diferença indica maior tendência a overfitting na configuração testada.

## 11. Avaliação final

No holdout de 500 pedidos, a regressão logística obteve:

- ROC-AUC: 0,877;
- Average Precision: 0,947;
- precisão de detratores: 91,5%;
- recall de detratores: 78,6%;
- F1 de detratores: 84,6%.

Foram identificados corretamente 291 dos 370 detratores existentes no holdout.

O modelo deixou de identificar 79 detratores e gerou 27 falsos alertas.

## 12. Importância por permutação

As principais variáveis foram:

1. quantidade de reclamações;
2. dias de atraso;
3. tempo de resolução.

A importância por permutação mede a perda de desempenho após embaralhar uma variável.

Não representa relação causal.

## 13. Leakage

Foram excluídas do modelo:

- `customer_id`;
- `order_id`;
- `nps_score`;
- `nps_category`;
- `is_detractor`;
- `repeat_purchase_30d`;
- `csat_internal_score`.

A recompra é uma informação posterior.

O momento de coleta do CSAT interno não está documentado.

O modelo pressupõe que o scoring ocorre após os dados operacionais de entrega e atendimento estarem disponíveis, porém antes da pesquisa de NPS.

## 14. Recomendações de negócio

A partir dos resultados, a empresa poderia avaliar:

1. criar alertas operacionais a partir de três dias de atraso;
2. priorizar pedidos com múltiplas reclamações;
3. escalonar clientes com contatos recorrentes;
4. acompanhar casos com maior tempo de resolução;
5. utilizar a pontuação do modelo para organizar uma fila de recuperação;
6. testar intervenções proativas através de experimentos controlados.

Essas ações são recomendações para investigação e teste.

A análise histórica não demonstra que essas intervenções causarão aumento do NPS.

## 15. Limitações

A base não possui:

- datas;
- SLA prometido;
- transportadora;
- categoria de produto;
- motivo do contato;
- devolução;
- taxa de resposta à pesquisa;
- custo das intervenções.

Também existem outras limitações:

- um pedido por cliente;
- ausência de validação temporal;
- ausência de teste prospectivo;
- ausência de experimento causal;
- probabilidades sem etapa específica de calibração;
- thresholds de risco ainda não calibrados segundo custo operacional.

## 16. Conclusão

A análise indica que a insatisfação observada está mais associada a fatores operacionais do que às características demográficas disponíveis na base.

Reclamações acumuladas e atrasos possuem destaque tanto na análise exploratória quanto na capacidade preditiva do modelo.

A regressão logística apresentou desempenho estável na validação cruzada e resultado semelhante no holdout, sendo escolhida como modelo final.

A proposta é utilizar o score como ferramenta de priorização de atendimento e recuperação, mantendo supervisão humana e monitoramento dos resultados.