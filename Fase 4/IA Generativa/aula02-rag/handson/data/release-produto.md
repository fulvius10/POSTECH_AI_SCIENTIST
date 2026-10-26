# Release Notes — InnovaAnalytics versão 8.2

## Destaques da versão

A versão 8.2 do InnovaAnalytics chega com foco em análise assistida por IA e
integração com o ecossistema de dados corporativo. Principais novidades:

- **Painel Executivo redesenhado**, com carregamento até 4x mais rápido e
  novos cartões de indicadores configuráveis.
- **Assistente de análise com linguagem natural**: o usuário pergunta em
  português ("qual filial mais vendeu no trimestre?") e recebe gráfico e
  resumo com as evidências utilizadas.
- **Conector nativo com Power BI e Google Looker**, com sincronização
  incremental a cada 15 minutos.
- **Exportação agendada** de relatórios em PDF e XLSX por e-mail.

## Melhorias

- Filtros de data agora aceitam períodos comparativos (ano contra ano,
  trimestre contra trimestre).
- Novo controle de permissões por pasta de relatório, herdado do diretório
  corporativo.
- Tela de administração exibe consumo de licenças em tempo real.

## Correções

- Corrigido o erro que impedia a abertura do relatório de vendas após a
  atualização 8.1.3 em ambientes com proxy corporativo.
- Corrigida a exportação para XLSX que truncava colunas com mais de 60
  caracteres.
- Corrigido o fuso horário incorreto em agendamentos criados no modo de
  horário de verão.

## Requisitos e compatibilidade

A versão 8.2 requer o InnovaServer 5.0 ou superior. Instalações locais devem
reservar 8 GB de RAM para o serviço de análise. Navegadores homologados:
Chrome, Edge e Firefox nas duas últimas versões estáveis.

## Suporte

Problemas com a versão 8.2 devem ser registrados no portal de suporte,
categoria "InnovaAnalytics", com o número da versão e evidências (prints ou
logs). O prazo de primeira resposta é de 8 horas úteis.
