import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

MODELO_OPENAI_PADRAO = os.environ.get("OPENAI_CHAT_MODEL", "")

INSTRUCAO_SISTEMA = (
    "Você é um assistente financeiro especialista. "
    "Use as ferramentas disponíveis para responder. "
    "Se a informação não vier de uma ferramenta, diga que não sabe."
)

# --- Ferramenta do vídeo 01 -------------------------------------------------

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
    # Em produção a query entra aqui — SEMPRE parametrizada, e a autorização
    # é checada AQUI contra o usuário da sessão, nunca confiando na filial
    # que o LLM decidiu passar.
    return f"Resultado da consulta para {filial} em {mes}/{ano}: R$ 1.250.000,00"

def criar_agente(llm):
    """O system prompt e o scratchpad são responsabilidade do create_agent."""
    return create_agent(
        llm,
        tools=[consultar_vendas],
        system_prompt=INSTRUCAO_SISTEMA,
    )

def executar_e_mostrar(agente, pergunta: str):
    """Imprime cada passo do loop — é o equivalente ao antigo verbose=True."""
    print(f"\n[Pergunta] {pergunta}\n")

    vistas = 0
    for estado in agente.stream({"messages": [HumanMessage(content=pergunta)]},
                                stream_mode="values"):
        mensagens = estado["messages"]
        for msg in mensagens[vistas:]:
            if msg.type == "ai" and msg.tool_calls:
                for chamada in msg.tool_calls:
                    print(f"  [decisão] chamar {chamada['name']} com {chamada['args']}")
            elif msg.type == "tool":
                print(f"  [ferramenta] {msg.content}")
            elif msg.type == "ai":
                print(f"\n[Resposta] {msg.content}")
        vistas = len(mensagens)

def run_aula_4_video_2():
    print("--- AULA 4 / VÍDEO 02: AGENTE COM TOOL CALLING ---")

    llm = ChatOpenAI(model=os.environ.get("OPENAI_CHAT_MODEL", MODELO_OPENAI_PADRAO))
    agente = criar_agente(llm)

    # Caso 1: a pergunta cabe na ferramenta.
    executar_e_mostrar(agente, "Quanto vendemos na filial São Paulo em Janeiro de 2026?")

    # Caso 2: filial fora da allowlist do vídeo 01. O Pydantic barra, o erro
    # volta ao agente como resultado da ferramenta e ele se corrige.
    executar_e_mostrar(agente, "E quanto vendemos na filial de Manaus em Janeiro de 2026?")

if __name__ == "__main__":
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERRO: configure OPENAI_API_KEY no .env antes de rodar o script.")
    else:
        run_aula_4_video_2()