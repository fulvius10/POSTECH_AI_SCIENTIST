import operator
import os
import time
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

ESPECIALISTAS = ("pesquisador", "analista", "redator")
MAX_PASSOS = 6  # trava contra supervisor indeciso girando para sempre

def criar_llm():
    return ChatOpenAI(model=os.environ.get("OPENAI_CHAT_MODEL", MODELO_OPENAI_PADRAO))

# ===========================================================================
# 1. Padrão Supervisor
# ===========================================================================

class EstadoEquipe(TypedDict):
    tarefa: str
    proximo: str
    passos: int
    entregas: Annotated[list[str], operator.add]
    trilha: Annotated[list[str], operator.add]

def criar_app_supervisor(llm):
    def supervisor(estado: EstadoEquipe) -> dict:
        """Decide quem trabalha agora. Não executa nada — só roteia."""
        ja_feito = [e.split(":")[0] for e in estado["entregas"]]

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Você coordena uma equipe. Especialistas disponíveis: {equipe}.\n"
             "Já entregaram: {feito}.\n"
             "Responda com UMA palavra: o próximo especialista, ou FINISH se a "
             "tarefa estiver completa. Só a palavra."),
            ("human", "Tarefa: {tarefa}"),
        ])
        bruto = (prompt | llm | StrOutputParser()).invoke({
            "equipe": ", ".join(ESPECIALISTAS),
            "feito": ", ".join(ja_feito) or "ninguém ainda",
            "tarefa": estado["tarefa"],
        }).strip().lower()

        # O supervisor é um LLM: valide a decisão dele contra a lista real.
        proximo = bruto if bruto in ESPECIALISTAS else "FINISH"

        return {
            "proximo": proximo,
            "passos": estado["passos"] + 1,
            "trilha": [f"supervisor -> {proximo}"],
        }

    def criar_especialista(nome: str, instrucao: str):
        def no(estado: EstadoEquipe) -> dict:
            prompt = ChatPromptTemplate.from_messages([
                ("system", instrucao + " Responda em uma frase."),
                ("human", "Tarefa: {tarefa}"),
            ])
            texto = (prompt | llm | StrOutputParser()).invoke(
                {"tarefa": estado["tarefa"]}
            )
            return {"entregas": [f"{nome}: {texto}"], "trilha": [nome]}
        return no

    def rotear(estado: EstadoEquipe) -> str:
        """ARESTA CONDICIONAL: lê a decisão do supervisor e devolve o nó."""
        if estado["passos"] >= MAX_PASSOS:
            return END
        if estado["proximo"] in ESPECIALISTAS:
            return estado["proximo"]
        return END

    grafo = StateGraph(EstadoEquipe)
    grafo.add_node("supervisor", supervisor)
    grafo.add_node("pesquisador",
                   criar_especialista("pesquisador", "Você levanta dados e fontes."))
    grafo.add_node("analista",
                   criar_especialista("analista", "Você interpreta números e aponta causas."))
    grafo.add_node("redator",
                   criar_especialista("redator", "Você escreve o texto final para a diretoria."))

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges("supervisor", rotear, [*ESPECIALISTAS, END])

    for nome in ESPECIALISTAS:
        grafo.add_edge(nome, "supervisor")

    return grafo.compile()

# ===========================================================================
# 2. Latência: sequencial vs paralelo
# ===========================================================================

class EstadoConsulta(TypedDict):
    tarefa: str
    fontes: Annotated[list[str], operator.add]


def _consulta_lenta(nome: str, segundos: float = 1.0):
    """Simula uma chamada de API ou LLM que demora."""
    def no(estado: EstadoConsulta) -> dict:
        time.sleep(segundos)
        return {"fontes": [nome]}
    return no

def criar_app_sequencial():
    grafo = StateGraph(EstadoConsulta)
    grafo.add_node("crm", _consulta_lenta("crm"))
    grafo.add_node("erp", _consulta_lenta("erp"))
    grafo.add_edge(START, "crm")
    grafo.add_edge("crm", "erp")      # erp só começa quando crm termina
    grafo.add_edge("erp", END)
    return grafo.compile()

def criar_app_paralelo():
    grafo = StateGraph(EstadoConsulta)
    grafo.add_node("crm", _consulta_lenta("crm"))
    grafo.add_node("erp", _consulta_lenta("erp"))
    grafo.add_edge(START, "crm")
    grafo.add_edge(START, "erp")      # duas arestas saindo do START = fan-out
    grafo.add_edge("crm", END)
    grafo.add_edge("erp", END)
    return grafo.compile()

def run_aula_5_video_3():
    print("--- AULA 5 / VÍDEO 03: MULTI-AGENTES E ENGENHARIA ---")

    llm = criar_llm()
    app = criar_app_supervisor(llm)

    print("\n=== 1. O desenho da equipe ===")
    print(app.get_graph().draw_mermaid())

    print("\n=== 2. O supervisor distribuindo o trabalho ===")
    entrada = {
        "tarefa": "Explicar a queda de 8% no faturamento do trimestre",
        "proximo": "", "passos": 0, "entregas": [], "trilha": [],
    }
    final = app.invoke(entrada)

    for passo in final["trilha"]:
        print(f"  {passo}")

    print("\n=== 3. O que cada especialista entregou ===")
    for entrega in final["entregas"]:
        print(f"  - {entrega}")

    print("\n=== 4. Latência: dois nós independentes ===")
    entrada_consulta = {"tarefa": "consolidar dados", "fontes": []}

    inicio = time.perf_counter()
    r1 = criar_app_sequencial().invoke(entrada_consulta)
    seq = time.perf_counter() - inicio

    inicio = time.perf_counter()
    r2 = criar_app_paralelo().invoke(entrada_consulta)
    par = time.perf_counter() - inicio

    print(f"  sequencial (crm -> erp): {seq:.2f}s  fontes={r1['fontes']}")
    print(f"  paralelo   (crm | erp):  {par:.2f}s  fontes={r2['fontes']}")
    print("\n  A diferença é só o desenho das arestas. Quando dois nós não")
    print("  dependem um do outro, ligue os dois ao mesmo nó anterior.")

if __name__ == "__main__":

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERRO: configure OPENAI_API_KEY no .env antes de rodar o script.")
    else:
        run_aula_5_video_3()