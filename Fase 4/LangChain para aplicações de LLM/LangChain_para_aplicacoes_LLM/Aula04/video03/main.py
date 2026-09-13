import logging
import os
from contextvars import ContextVar
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger('api-financeira')

DEBUG_ERROS = os.environ.get("DEBUG_ERROS", "0").lower() in {"1", "true", "sim"}

MODELO_OPENAI_PADRAO = os.environ.get("OPENAI_CHAT_MODEL", "")

INSTRUCAO_SISTEMA = (
    "Você é um assistente financeiro especialista. "
    "Use as ferramentas disponíveis para responder. "
    "Se a informação não vier de uma ferramenta, diga que não sabe."
)

ORIGENS_PERMITIDAS = os.environ.get("CORS_ORIGENS", "http://localhost:3000").split(",")

# --- Identidade do chamador -------------------------------------------------
# O agente roda dentro do request; a ferramenta precisa saber QUEM perguntou.
# Um ContextVar carrega isso até lá sem passar pelo prompt (prompt não é
# controle de acesso — o LLM pode ignorar qualquer instrução de permissão).

usuario_atual: ContextVar[str] = ContextVar("usuario_atual")

esquema_api_key = APIKeyHeader(name="x-api-key", auto_error=False)

# Em produção isto vem do banco/IdP. Aqui é só para a aula.
FILIAIS_POR_USUARIO = {
    "analista_sp": {"Geral", "São Paulo"},
    "diretoria": {"Geral", "São Paulo", "Recife", "Curitiba"},
}

def autenticar(x_api_key: str = Security(esquema_api_key)) -> str:
    """Troca a chave do header por uma identidade. Deny-by-default."""
    usuario = {
        os.environ.get("API_KEY_ANALISTA", ""): "analista_sp",
        os.environ.get("API_KEY_DIRETORIA", ""): "diretoria",
    }.get(x_api_key)

    if not x_api_key or not usuario:
        raise HTTPException(status_code=401, detail="Credencial inválida.")
    return usuario

# --- Ferramenta -------------------------------------------------------------

MESES = Literal[
    "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro",
]

FILIAIS = Literal["Geral", "São Paulo", "Recife", "Curitiba"]


class ConsultarVendasSchema(BaseModel):
    mes: MESES = Field(..., description="Mês de referência")
    ano: int = Field(..., ge=2020, le=2030, description="Ano com 4 dígitos")
    filial: FILIAIS = Field("Geral", description="Nome da filial para filtrar")


@tool("consultar_vendas_db", args_schema=ConsultarVendasSchema)
def consultar_vendas(mes: str, ano: int, filial: str = "Geral"):
    """Consulta o total de vendas de uma filial em um mês específico.

    Use quando o usuário pedir faturamento, vendas ou performance de um período.
    """
    # A autorização é checada AQUI, contra o usuário do request — não contra o
    # que o LLM decidiu pedir. Devolver texto em vez de erro deixa o agente
    # explicar a recusa ao usuário.
    permitidas = FILIAIS_POR_USUARIO.get(usuario_atual.get(""), set())
    if filial not in permitidas:
        return f"Acesso negado: este usuário não pode consultar a filial {filial}."

    # Em produção, query parametrizada:
    #   cursor.execute("SELECT SUM(valor) FROM vendas WHERE mes=%s AND ano=%s AND filial=%s",
    #                  (mes, ano, filial))
    return f"Resultado da consulta para {filial} em {mes}/{ano}: R$ 1.250.000,00"

# --- Agente -----------------------------------------------------------------

def criar_agente(llm=None):
    llm = llm or ChatOpenAI(model=os.environ.get("OPENAI_CHAT_MODEL", MODELO_OPENAI_PADRAO))
    return create_agent(llm, tools=[consultar_vendas], system_prompt=INSTRUCAO_SISTEMA)

# --- Contrato da API --------------------------------------------------------

class PerguntaRequest(BaseModel):
    pergunta: str = Field(..., min_length=3, max_length=500)


class RespostaAnalise(BaseModel):
    resposta: str
    ferramentas_usadas: list[str]
    # Só preenchido com DEBUG_ERROS=1. Mostra COM QUE ARGUMENTOS o LLM decidiu
    # chamar cada ferramenta — é o que revela se ele normalizou a entrada.
    detalhe_chamadas: list[dict] | None = None


app = FastAPI(title="API Financeira Agêntica", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENS_PERMITIDAS,
    allow_methods=["POST"],
    allow_headers=["x-api-key", "content-type"],
)

agente = criar_agente()

@app.post("/analise-financeira", response_model=RespostaAnalise)
async def analise_financeira(
    corpo: PerguntaRequest,
    usuario: str = Depends(autenticar),
) -> RespostaAnalise:
    token = usuario_atual.set(usuario)
    try:
        estado = await agente.ainvoke(
            {"messages": [HumanMessage(content=corpo.pergunta)]}
        )
    except Exception as e:
        # O traceback completo vai para o console do uvicorn.
        logger.exception("Falha ao invocar o agente (usuário=%s)", usuario)
        detalhe = (f"{type(e).__name__}: {e}" if DEBUG_ERROS
                   else "Falha ao consultar o assistente.")
        raise HTTPException(status_code=502, detail=detalhe)
    finally:
        usuario_atual.reset(token)
 
    mensagens = estado["messages"]
    chamadas = [
        chamada
        for msg in mensagens
        if msg.type == "ai" and msg.tool_calls
        for chamada in msg.tool_calls
    ]
    return RespostaAnalise(
        resposta=mensagens[-1].content,
        ferramentas_usadas=[c["name"] for c in chamadas],
        detalhe_chamadas=(
            [{"ferramenta": c["name"], "argumentos": c["args"]} for c in chamadas]
            if DEBUG_ERROS else None
        ),
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)