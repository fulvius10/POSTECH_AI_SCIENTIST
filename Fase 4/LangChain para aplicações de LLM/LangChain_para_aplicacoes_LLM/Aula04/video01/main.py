"""
1. O schema não é documentação — ele REJEITA entrada inválida em tempo de execução.
2. O schema também é o contrato que o LLM enxerga: restrição no schema reduz
   alucinação, porque o modelo vê os valores permitidos antes de chamar.
3. Os argumentos vêm do LLM, que os inventou a partir de texto do usuário.
   Isso é entrada não confiável chegando no seu banco.
"""

import json
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

MESES = Literal[
    "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro",
]

FILIAIS = Literal["Geral", "São Paulo", "Recife", "Curitiba"]

class ConsultarVendasSchema(BaseModel):
    """Esquema de entrada com validação rigorosa."""

    mes: MESES = Field(..., description="Mês de referência")
    ano: int = Field(..., ge=2020, le=2030, description="Ano com 4 dígitos")
    filial: FILIAIS = Field("Geral", description="Nome da filial para filtrar")

@tool("consultar_vendas_db", args_schema=ConsultarVendasSchema)
def consultar_vendas(mes: str, ano: int, filial: str = "Geral"):
    """Consulta o total de vendas de uma filial em um mês específico.

    Use quando o usuário pedir faturamento, vendas ou performance de um período.
    Não use para projeções futuras nem para despesas.
    """
    # Em produção a query entra aqui — SEMPRE parametrizada:
    #   cursor.execute(
    #       "SELECT SUM(valor) FROM vendas WHERE mes = %s AND ano = %s AND filial = %s",
    #       (mes, ano, filial),
    #   )
    # E a autorização é checada AQUI, comparando com o usuário da sessão —
    # nunca confiando no que o LLM decidiu passar como filial.
    return f"Resultado da consulta para {filial} em {mes}/{ano}: R$ 1.250.000,00"

def demonstrar():
    print("=== O que o LLM enxerga (JSON Schema gerado) ===")
    print(json.dumps(consultar_vendas.args, ensure_ascii=False, indent=2))

    print("\n=== Chamada válida ===")
    print(consultar_vendas.invoke({"mes": "Março", "ano": 2026}))

    print("\n=== Chamadas que o schema rejeita ===")
    invalidas = [
        {"mes": "banana", "ano": 2026},                              # mês inexistente
        {"mes": "Março", "ano": "abcd"},                             # ano não numérico
        {"mes": "Março", "ano": 1800},                               # fora da faixa
        {"mes": "Março", "ano": 2026, "filial": "'; DROP TABLE--"},  # fora da allowlist
    ]
    for argumentos in invalidas:
        try:
            print("ACEITOU (não deveria):", consultar_vendas.invoke(argumentos))
        except Exception as e:
            print(f"rejeitado -> {argumentos}")
            print(f"   {type(e).__name__}")

if __name__ == "__main__":
    demonstrar()