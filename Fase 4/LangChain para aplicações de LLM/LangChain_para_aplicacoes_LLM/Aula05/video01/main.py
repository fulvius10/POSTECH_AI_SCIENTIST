import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

MODELO_OPENAI_PADRAO = os.environ.get("OPENAI_CHAT_MODEL")
if not MODELO_OPENAI_PADRAO:
    raise RuntimeError("Configure OPENAI_CHAT_MODEL no .env")

CATEGORIAS = ("financeiro", "tecnico", "comercial")

def criar_llm():
    return ChatOpenAI(model=os.environ.get("OPENAI_CHAT_MODEL", MODELO_OPENAI_PADRAO))

# ===========================================================================
# 1. O jeito linear (LCEL) — o que as aulas anteriores usaram
# ===========================================================================

def executar_lcel(llm, pergunta: str) -> str:
    """Input -> LLM -> Parse -> Output. Uma via só, sem estado."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um atendente corporativo. Responda em uma frase."),
        ("human", "{pergunta}"),
    ])
    return (prompt | llm | StrOutputParser()).invoke({"pergunta": pergunta})

# ===========================================================================
# 2. O mesmo problema como máquina de estados
# ===========================================================================

class EstadoAtendimento(TypedDict):
    """O contrato de dados do grafo.

    Repare que ele carrega muito mais que 'mensagens': categoria é uma decisão
    intermediária, e trilha é a auditoria de tudo que aconteceu. Num caso real
    entrariam aqui DataFrames, flags de compliance, id do cliente etc.
    """

    pergunta: str
    categoria: str
    resposta: str
    # Annotated + operator.add = REDUCER: o LangGraph SOMA as listas devolvidas
    # por cada nó em vez de sobrescrever. É como o histórico se acumula.
    trilha: Annotated[list[str], operator.add]

def criar_app(llm):
    def classificar(estado: EstadoAtendimento) -> dict:
        """Nó 1: decide a categoria. Devolve SÓ o que mudou."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Classifique a pergunta em exatamente uma palavra, entre: "
             f"{', '.join(CATEGORIAS)}. Responda só a palavra."),
            ("human", "{pergunta}"),
        ])
        bruto = (prompt | llm | StrOutputParser()).invoke(
            {"pergunta": estado["pergunta"]}
        ).strip().lower()

        # O LLM é probabilístico: valide a saída antes de confiar nela.
        categoria = bruto if bruto in CATEGORIAS else "comercial"

        return {"categoria": categoria, "trilha": [f"classificar -> {categoria}"]}

    def responder(estado: EstadoAtendimento) -> dict:
        """Nó 2: usa a decisão do nó anterior, que veio pelo estado."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Você é um especialista da área {categoria}. Responda em uma frase."),
            ("human", "{pergunta}"),
        ])
        texto = (prompt | llm | StrOutputParser()).invoke({
            "categoria": estado["categoria"],
            "pergunta": estado["pergunta"],
        })
        return {"resposta": texto, "trilha": ["responder"]}

    def auditar(estado: EstadoAtendimento) -> dict:
        """Nó 3: nem todo nó precisa de LLM. Aqui é Python puro."""
        return {"trilha": [f"auditar -> resposta com {len(estado['resposta'])} chars"]}

    grafo = StateGraph(EstadoAtendimento)
    grafo.add_node("classificar", classificar)
    grafo.add_node("responder", responder)
    grafo.add_node("auditar", auditar)

    grafo.add_edge(START, "classificar")
    grafo.add_edge("classificar", "responder")
    grafo.add_edge("responder", "auditar")
    grafo.add_edge("auditar", END)

    return grafo.compile()

def run_aula_5_video_1():
    print("--- AULA 5 / VÍDEO 01: DA CHAIN LINEAR À MÁQUINA DE ESTADOS ---")

    llm = criar_llm()
    pergunta = "Nosso faturamento caiu 8% no trimestre. O que devo olhar primeiro?"

    print("\n=== 1. LCEL: entra string, sai string ===")
    print(f"Pergunta: {pergunta}")
    print(f"Resposta: {executar_lcel(llm, pergunta)}")
    print("\nNão sobrou nada: nenhuma decisão intermediária, nenhum rastro.")

    app = criar_app(llm)

    print("\n=== 2. O desenho do grafo ===")
    print(app.get_graph().draw_mermaid())
    print("Cole isso em https://mermaid.live para ver o diagrama.")

    print("\n=== 3. StateGraph: entra estado, sai estado ===")
    estado_final = app.invoke({
        "pergunta": pergunta,
        "categoria": "",
        "resposta": "",
        "trilha": [],
    })

    print(f"Categoria decidida: {estado_final['categoria']}")
    print(f"Resposta:           {estado_final['resposta']}")
    print("\nTrilha acumulada pelo reducer:")
    for passo in estado_final["trilha"]:
        print(f"  - {passo}")

    print("\n=== 4. Vendo o estado a cada passo ===")
    for parcial in app.stream(
        {"pergunta": pergunta, "categoria": "", "resposta": "", "trilha": []},
        stream_mode="updates",
    ):
        for no, mudanca in parcial.items():
            print(f"  [{no}] devolveu: {list(mudanca.keys())}")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERRO: configure OPENAI_API_KEY no .env antes de rodar o script.")
    else:
        run_aula_5_video_1()